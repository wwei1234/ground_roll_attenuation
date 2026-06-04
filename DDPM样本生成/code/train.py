# diffusion_train.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import SeismicPix2PixDataset
from model import DiffusionUNet, GaussianDiffusion

# ==================== 用户配置区 ====================
GATHER_DIR = r"D:\桌面\面波压制\深层数据面波压制\原始炮集"
MASK_DIR = r"D:\桌面\面波压制\深层数据面波压制\masks"
CHECKPOINT_DIR = "./checkpoints_diffusion"
LOG_DIR = "./logs_diffusion"

BATCH_SIZE = 2
NUM_EPOCHS = 200
LR = 1e-4

# 扩散模型配置
TIMESTEPS = 1000
BETA_SCHEDULE = 'cosine'   # 'cosine' 通常比 'linear' 生成质量更稳定
DDIM_STEPS = 50            # 验证/生成时用的快速步数

# 混合损失权重（在噪声 MSE 基础上，额外约束预测的 x0）
LAMBDA_L1 = 1.0      # x0 L1
LAMBDA_FREQ = 1.0    # 频谱损失
LAMBDA_TRACE = 5.0   # 同相轴连续性

VAL_RATIO = 0.1
SAVE_INTERVAL = 10
VISUALIZE_EVERY = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================================


def save_checkpoint(epoch, model, opt, sched, save_dir):
    path = os.path.join(save_dir, f"diffusion_epoch_{epoch:04d}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'sched_state_dict': sched.state_dict(),
    }, path)
    print(f"  [Checkpoint] 已保存: {path}")


def load_checkpoint(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    opt.load_state_dict(ckpt['opt_state_dict'])
    sched.load_state_dict(ckpt['sched_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  [Checkpoint] 已加载: {path}，从 Epoch {start_epoch} 继续")
    return start_epoch


def visualize_batch(writer, tag, mask, real, fake, epoch, max_samples=4):
    """将 mask / 真实炮集 / 生成炮集并排写入 TensorBoard"""
    n = min(max_samples, mask.shape[0])
    to01 = lambda t: (t.clamp(-1, 1) + 1) / 2
    rows = [torch.cat([to01(mask[i]), to01(real[i]), to01(fake[i])], dim=2)
            for i in range(n)]
    grid = torch.cat(rows, dim=1)
    writer.add_image(tag, grid, global_step=epoch)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # ── 数据集（完全复用现有 dataset.py） ──
    full_dataset = SeismicPix2PixDataset(GATHER_DIR, MASK_DIR, augment=True)

    n_val = max(1, int(len(full_dataset) * VAL_RATIO))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"训练集: {n_train} 对  |  验证集: {n_val} 对  |  设备: {DEVICE}")
    print(f"扩散配置: T={TIMESTEPS}, schedule={BETA_SCHEDULE}, DDIM_steps={DDIM_STEPS}")
    print(f"混合损失: L1={LAMBDA_L1}, Freq={LAMBDA_FREQ}, Trace={LAMBDA_TRACE}")

    # ── 模型与扩散过程 ──
    model = DiffusionUNet(in_channels=2, out_channels=1, base_ch=48).to(DEVICE)
    diffusion = GaussianDiffusion(timesteps=TIMESTEPS, beta_schedule=BETA_SCHEDULE)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型参数量: {n_params:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

    # ── 断点续训 ──
    start_epoch = 1
    ckpt_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])
    if ckpt_files:
        latest = os.path.join(CHECKPOINT_DIR, ckpt_files[-1])
        start_epoch = load_checkpoint(latest, model, opt, sched, DEVICE)

    # ── 训练主循环 ──
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0

        for batch in train_loader:
            gather = batch['B'].to(DEVICE)  # 目标炮集 (B,1,H,W)
            mask = batch['A'].to(DEVICE)    # 条件 mask (B,1,H,W)

            # 随机采样时间步
            t = torch.randint(0, TIMESTEPS, (gather.size(0),), device=DEVICE).long()

            # 计算损失（噪声 MSE + 可选 x0 物理约束）
            loss = diffusion.p_losses(
                model, gather, mask, t,
                lambda_l1=LAMBDA_L1,
                lambda_freq=LAMBDA_FREQ,
                lambda_trace=LAMBDA_TRACE
            )

            opt.zero_grad()
            loss.backward()
            # 梯度裁剪，扩散模型训练稳定的关键
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item()

        sched.step()

        # ── 验证：用 DDIM 快速生成，计算 Val L1 ──
        model.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mask_v = batch['A'].to(DEVICE)
                real_v = batch['B'].to(DEVICE)

                # DDIM 快速采样（50步），无需梯度
                fake_v = diffusion.ddim_sample(
                    model, real_v.shape, mask_v, DEVICE,
                    ddim_timesteps=DDIM_STEPS, eta=0.0
                )
                val_l1 += F.l1_loss(fake_v, real_v).item()

        val_l1 /= len(val_loader)
        avg_loss = loss_sum / len(train_loader)

        print(f"Epoch [{epoch:4d}/{NUM_EPOCHS}]  "
              f"Loss: {avg_loss:.4f}  Val_L1: {val_l1:.4f}  "
              f"LR: {sched.get_last_lr()[0]:.2e}")

        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Val_L1', val_l1, epoch)
        writer.add_scalar('LR', sched.get_last_lr()[0], epoch)

        # ── 可视化 ──
        if epoch % VISUALIZE_EVERY == 0:
            with torch.no_grad():
                sample = next(iter(val_loader))
                mask_s = sample['A'][:4].to(DEVICE)
                real_s = sample['B'][:4].to(DEVICE)
                fake_s = diffusion.ddim_sample(
                    model, real_s.shape, mask_s, DEVICE,
                    ddim_timesteps=DDIM_STEPS, eta=0.0
                )
            visualize_batch(writer, 'Samples/mask|real|fake',
                            mask_s.cpu(), real_s.cpu(), fake_s.cpu(), epoch)

        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            save_checkpoint(epoch, model, opt, sched, CHECKPOINT_DIR)

    writer.close()
    print(f"\n训练完成！Checkpoint: {CHECKPOINT_DIR}")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR}")


if __name__ == "__main__":
    train()