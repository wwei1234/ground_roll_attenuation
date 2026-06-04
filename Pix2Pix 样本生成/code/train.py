# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import SeismicPix2PixDataset
from model import build_unet_generator, PatchGANDiscriminator, init_weights

# ==================== 用户配置区 ====================
GATHER_DIR     = r"D:\桌面\面波压制\深层数据面波压制\原始炮集"
MASK_DIR       = r"D:\桌面\面波压制\深层数据面波压制\masks"
CHECKPOINT_DIR = "./checkpoints_pix2pix_6_4"
LOG_DIR        = "./logs_pix2pix_6_4"

# 训练超参数
BATCH_SIZE = 4
NUM_EPOCHS = 200
LR         = 1e-4    # 原版 2e-4 对 L1 权重过高时容易模糊，降至 1e-4
BETA1      = 0.5

# ── 损失权重 ────────────────────────────────────────
# LAMBDA_L1：像素域重建损失。原版推荐 100，但过高会导致生成结果模糊（生成器
#   倾向于输出"均值"而非清晰纹理）。降至 20 后 GAN 损失话语权更大，纹理更清晰。
LAMBDA_L1    = 20.0

# LAMBDA_FM：Feature Matching Loss 权重。让生成器在判别器的各中间层特征上
#   与真实炮集对齐，弥补 L1 在像素域过于局部的问题，有效减少模糊。
#   推荐范围 5–20，此处设为 10。
LAMBDA_FM    = 10.0

# LAMBDA_FREQ：低频幅度谱 L1 损失，约束面波的频率域特征。
LAMBDA_FREQ  = 1.0

# LAMBDA_TRACE：同相轴连续性损失，惩罚相邻道间差分不一致，
#   鼓励生成炮集的同相轴沿道方向连续。推荐范围 10–50。
LAMBDA_TRACE = 20.0

VAL_RATIO        = 0.1
SAVE_INTERVAL    = 10
VISUALIZE_EVERY  = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================================

# ─────────────────────────────────────────────────────────────
#  损失函数
# ─────────────────────────────────────────────────────────────

def frequency_loss(real, fake):
    """
    低频段幅度谱 L1 损失（沿时间轴 dim=-2 做 FFT，取前 20% 低频分量）。
    面波能量集中于低频，此损失防止生成器忽略面波的频率特征。
    """
    def low_freq_amp(x):
        spec  = torch.fft.rfft(x, dim=-2)
        amp   = torch.abs(spec)
        n_low = max(1, amp.shape[-2] // 5)
        return amp[..., :n_low, :]
    return nn.functional.l1_loss(low_freq_amp(fake), low_freq_amp(real))


def feature_matching_loss(feats_fake, feats_real):
    """
    Feature Matching Loss：生成器在判别器各中间层的特征与真实炮集对齐。

    原理：判别器各层特征是对输入的多尺度抽象表示。让生成结果在这些抽象
    层面上接近真实数据，比单纯在像素层面接近（L1）更符合感知相似度，
    能有效减少生成结果的模糊感。

    注意：真实特征必须 detach()，避免通过判别器的参数影响生成器梯度，
    即只优化生成器，不改变判别器对"真实"的判断标准。
    """
    loss = 0.0
    for ff, fr in zip(feats_fake, feats_real):
        loss += nn.functional.l1_loss(ff, fr.detach())
    return loss / len(feats_fake)


def trace_continuity_loss(fake, real):
    """
    同相轴连续性损失：惩罚生成炮集与真实炮集在道间差分上的不一致。

    面波是空间上连续的线性同相轴，此损失在损失函数层面显式编码这一
    物理先验，促使生成器输出道间平滑连续的波形，而非孤立的道间噪声。

    fake, real: (B, 1, H, W)，W 为道数方向（axis=-1）
    """
    diff_real = real[..., 1:] - real[..., :-1]   # 真实炮集道间差分
    diff_fake = fake[..., 1:] - fake[..., :-1]   # 生成炮集道间差分
    return nn.functional.l1_loss(diff_fake, diff_real)

# ─────────────────────────────────────────────────────────────
#  Checkpoint 工具
# ─────────────────────────────────────────────────────────────

def save_checkpoint(epoch, G, D, opt_G, opt_D, save_dir):
    path = os.path.join(save_dir, f"pix2pix_epoch_{epoch:04d}.pth")
    torch.save({
        'epoch':            epoch,
        'G_state_dict':     G.state_dict(),
        'D_state_dict':     D.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
    }, path)
    print(f"  [Checkpoint] 已保存: {path}")

def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    D.load_state_dict(ckpt['D_state_dict'])
    opt_G.load_state_dict(ckpt['opt_G_state_dict'])
    opt_D.load_state_dict(ckpt['opt_D_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  [Checkpoint] 已加载: {path}，从 Epoch {start_epoch} 继续训练")
    return start_epoch

def visualize_batch(writer, tag, mask, real, fake, epoch, max_samples=4):
    """将 mask / 真实炮集 / 生成炮集并排写入 TensorBoard（每行一个样本）"""
    n    = min(max_samples, mask.shape[0])
    to01 = lambda t: (t.clamp(-1, 1) + 1) / 2
    rows = [torch.cat([to01(mask[i]), to01(real[i]), to01(fake[i])], dim=2)
            for i in range(n)]
    grid = torch.cat(rows, dim=1)
    writer.add_image(tag, grid, global_step=epoch)


# ─────────────────────────────────────────────────────────────
#  训练主函数
# ─────────────────────────────────────────────────────────────

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # ── 数据集 ───────────────────────────────────────────────
    # 直接传入 data_dir / label_dir，不再需要 TARGET_SHAPE（不做 resize）
    full_dataset = SeismicPix2PixDataset(GATHER_DIR, MASK_DIR, augment=True)

    n_val   = max(1, int(len(full_dataset) * VAL_RATIO))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"训练集: {n_train} 对  |  验证集: {n_val} 对  |  设备: {DEVICE}")
    print(f"损失权重：L1={LAMBDA_L1}, FM={LAMBDA_FM}, Freq={LAMBDA_FREQ}, Trace={LAMBDA_TRACE}")

    # ── 网络 ────────────────────────────────────────────────
    G = init_weights(build_unet_generator(in_channels=1, out_channels=1, n_down=7).to(DEVICE))
    D = init_weights(PatchGANDiscriminator(in_channels=2).to(DEVICE))

    # ── 损失函数 ─────────────────────────────────────────────
    criterion_gan = nn.MSELoss()   # LSGAN，比 BCE 更稳定
    criterion_l1  = nn.L1Loss()

    # ── 优化器 ───────────────────────────────────────────────
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    decay_start = NUM_EPOCHS // 2
    lr_lambda   = lambda ep: 1.0 - max(0, ep - decay_start) / float(NUM_EPOCHS - decay_start + 1)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # ── 断点续训 ─────────────────────────────────────────────
    start_epoch = 1
    ckpt_files  = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])
    if ckpt_files:
        latest = os.path.join(CHECKPOINT_DIR, ckpt_files[-1])
        start_epoch = load_checkpoint(latest, G, D, opt_G, opt_D, DEVICE)

    # ── 训练主循环 ──────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        G.train()
        D.train()

        loss_G_sum = loss_D_sum = 0.0

        for batch in train_loader:
            mask_real   = batch['A'].to(DEVICE)   # 条件：mask，(B,1,H,W)
            gather_real = batch['B'].to(DEVICE)   # 目标：真实炮集，(B,1,H,W)

            gather_fake = G(mask_real)

            # ══════════════════════════════════════════════
            #  步骤 1：更新判别器 D
            # ══════════════════════════════════════════════
            opt_D.zero_grad()

            pred_real = D(mask_real, gather_real)
            pred_fake = D(mask_real, gather_fake.detach())

            loss_D = (criterion_gan(pred_real, torch.ones_like(pred_real)) +
                      criterion_gan(pred_fake, torch.zeros_like(pred_fake))) * 0.5
            loss_D.backward()
            opt_D.step()

            # ══════════════════════════════════════════════
            #  步骤 2：更新生成器 G
            #
            #  损失组成（四项）：
            #    1. GAN 损失：欺骗判别器
            #    2. L1 损失：像素域重建（权重已降至 20，避免过度模糊）
            #    3. Feature Matching Loss：判别器中间层特征对齐（新增）
            #    4. 频谱损失：低频面波能量分布对齐
            #    5. 同相轴连续性损失：道间差分对齐（新增）
            # ══════════════════════════════════════════════
            opt_G.zero_grad()

            # 同时获取判别器评分和中间层特征
            pred_fake_G, feats_fake = D(mask_real, gather_fake,   return_features=True)
            _,           feats_real = D(mask_real, gather_real,   return_features=True)

            loss_G_gan   = criterion_gan(pred_fake_G, torch.ones_like(pred_fake_G))
            loss_G_l1    = criterion_l1(gather_fake, gather_real)       * LAMBDA_L1
            loss_G_fm    = feature_matching_loss(feats_fake, feats_real) * LAMBDA_FM
            loss_G_freq  = frequency_loss(gather_real, gather_fake)     * LAMBDA_FREQ
            loss_G_trace = trace_continuity_loss(gather_fake, gather_real) * LAMBDA_TRACE

            loss_G = loss_G_gan + loss_G_l1 + loss_G_fm + loss_G_freq + loss_G_trace
            loss_G.backward()
            opt_G.step()

            loss_G_sum += loss_G.item()
            loss_D_sum += loss_D.item()

        sched_G.step()
        sched_D.step()

        # ── 验证集评估 ─────────────────────────────────────
        G.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mask_v  = batch['A'].to(DEVICE)
                real_v  = batch['B'].to(DEVICE)
                fake_v  = G(mask_v)
                val_l1 += criterion_l1(fake_v, real_v).item()
        val_l1 /= len(val_loader)

        # ── 日志 ───────────────────────────────────────────
        n_batches  = len(train_loader)
        avg_loss_G = loss_G_sum / n_batches
        avg_loss_D = loss_D_sum / n_batches

        print(f"Epoch [{epoch:4d}/{NUM_EPOCHS}]  "
              f"Loss_G: {avg_loss_G:.4f}  Loss_D: {avg_loss_D:.4f}  "
              f"Val_L1: {val_l1:.4f}  LR: {sched_G.get_last_lr()[0]:.2e}")

        writer.add_scalar('Loss/Generator',     avg_loss_G, epoch)
        writer.add_scalar('Loss/Discriminator', avg_loss_D, epoch)
        writer.add_scalar('Loss/Val_L1',        val_l1,     epoch)
        writer.add_scalar('LR',                 sched_G.get_last_lr()[0], epoch)

        # ── TensorBoard 可视化 ─────────────────────────────
        if epoch % VISUALIZE_EVERY == 0:
            with torch.no_grad():
                sample  = next(iter(val_loader))
                mask_s  = sample['A'].to(DEVICE)
                real_s  = sample['B'].to(DEVICE)
                fake_s  = G(mask_s)
            visualize_batch(writer, 'Samples/mask|real|fake',
                            mask_s.cpu(), real_s.cpu(), fake_s.cpu(), epoch)

        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            save_checkpoint(epoch, G, D, opt_G, opt_D, CHECKPOINT_DIR)

    writer.close()
    print(f"\n训练完成！Checkpoint 保存于: {CHECKPOINT_DIR}")
    print(f"TensorBoard 日志目录: {LOG_DIR}")
    print(f"  运行 tensorboard --logdir={LOG_DIR} 查看训练曲线")


if __name__ == "__main__":
    train()