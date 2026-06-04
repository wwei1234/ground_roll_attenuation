import os
import csv
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from U_Net import UNet
from dataset import ManualMaskSeismicDataset

# ============================================================
#  ★ 超参数配置（修改这里）
# ============================================================
CONFIG = {
    # 训练基本参数
    "epochs"          : 200,
    "batch_size"      : 4,
    "learning_rate"   : 1e-4,
    "val_ratio"       : 0.2,
    "num_workers"     : 0,
    "seed"            : 42,

    # 混合精度（仅 CUDA 生效，CPU 自动关闭）
    "use_amp"         : True,

    # 模型保存 & 日志
    "save_dir"        : "U-Net_3_7",
    "log_dir"         : "logs_3_7",
    "save_every"      : 10,

    # 设备
    "device"          : "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================
#  ★ 多组 Dataset 配置
# ============================================================
DATASET_CONFIGS = [
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.05, 0.08),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "half_line",
    ),
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.05, 0.08),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "full_line",
    ),
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.08, 0.12),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "full_line",
    ),
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.08, 0.12),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "half_line",
    ),
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.12, 0.16),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "full_line",
    ),
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.12, 0.16),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "half_line",
    ),
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.16, 0.20),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "half_line",
    ),
    dict(
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.16, 0.20),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = "full_line",
    ),
]

# ============================================================
#  工具函数
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_l1_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """仅在 mask=1 的区域计算 L1 损失。"""
    diff   = torch.abs(pred - target)
    masked = diff * mask
    denom  = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def build_datasets() -> ConcatDataset:
    datasets = []
    for i, cfg in enumerate(DATASET_CONFIGS):
        print(f"\n[数据集 {i+1}/{len(DATASET_CONFIGS)}] 正在构建...")
        ds = ManualMaskSeismicDataset(**cfg)
        datasets.append(ds)
        print(f"  → 样本数: {len(ds)}")
    merged = ConcatDataset(datasets)
    print(f"\n合并后总样本数: {len(merged)}")
    return merged


def collate_fn(batch):
    """将 list of dict 整理成 dict of tensor，并添加 channel 维度。"""
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], np.ndarray):
            t = torch.from_numpy(np.stack(vals, axis=0))
            if k in ('input_data', 'label_data', 'original_data'):
                t = t.unsqueeze(1)
            elif k in ('loss_region', 'surface_wave_mask',
                       'non_sw_mask_positions', 'non_sw_region'):
                t = t.unsqueeze(1).float()
            out[k] = t
        else:
            out[k] = vals
    return out


def save_checkpoint(model, optimizer, scaler, epoch, val_loss, path):
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict'   : scaler.state_dict() if scaler else None,
        'val_loss'            : val_loss,
    }, path)


def init_log(log_path: str):
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'epoch_time_s', 'lr'])


def append_log(log_path: str, row: list):
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
#  训练 / 验证 单轮
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, epoch, total_epochs):
    model.train()
    total_loss  = 0.0
    running_loss = 0.0

    pbar = tqdm(
        loader,
        desc=f"Epoch [{epoch:4d}/{total_epochs}] 训练",
        ncols=100,
        leave=False,
        dynamic_ncols=False,
    )

    for step, batch in enumerate(pbar, 1):
        inp   = batch['input_data'].to(device)
        label = batch['label_data'].to(device)
        mask  = batch['loss_region'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            pred = model(inp)
            loss = masked_l1_loss(pred, label, mask)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss   += loss.item()
        running_loss  = total_loss / step
        pbar.set_postfix(loss=f"{running_loss:.6f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, use_amp, epoch, total_epochs):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(
        loader,
        desc=f"Epoch [{epoch:4d}/{total_epochs}] 验证",
        ncols=100,
        leave=False,
        dynamic_ncols=False,
    )

    for step, batch in enumerate(pbar, 1):
        inp   = batch['input_data'].to(device)
        label = batch['label_data'].to(device)
        mask  = batch['loss_region'].to(device)

        with autocast(enabled=use_amp):
            pred = model(inp)
            loss = masked_l1_loss(pred, label, mask)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / step:.6f}")

    return total_loss / len(loader)


# ============================================================
#  主训练流程
# ============================================================

def main():
    set_seed(CONFIG['seed'])
    device  = torch.device(CONFIG['device'])
    use_amp = CONFIG['use_amp'] and (CONFIG['device'] == 'cuda')

    print(f"\n使用设备  : {device}")
    print(f"混合精度  : {'开启 (AMP)' if use_amp else '关闭'}")

    # ---------- 目录 ----------
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'],  exist_ok=True)
    log_path  = os.path.join(CONFIG['log_dir'],  'train_log.csv')
    best_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
    init_log(log_path)

    # ---------- 数据集 ----------
    full_dataset = build_datasets()
    total_len    = len(full_dataset)
    val_len      = int(total_len * CONFIG['val_ratio'])
    train_len    = total_len - val_len

    train_set, val_set = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    print(f"\n训练集: {train_len} 样本  |  验证集: {val_len} 样本")

    train_loader = DataLoader(
        train_set,
        batch_size  = CONFIG['batch_size'],
        shuffle     = True,
        num_workers = CONFIG['num_workers'],
        collate_fn  = collate_fn,
        pin_memory  = (CONFIG['device'] == 'cuda'),
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = CONFIG['batch_size'],
        shuffle     = False,
        num_workers = CONFIG['num_workers'],
        collate_fn  = collate_fn,
        pin_memory  = (CONFIG['device'] == 'cuda'),
    )

    # ---------- 模型 & 优化器 ----------
    model     = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    scaler = GradScaler(enabled=use_amp)

    # ---------- 训练循环 ----------
    best_val_loss = float('inf')
    total_epochs  = CONFIG['epochs']

    print("\n" + "=" * 65)
    print(f"开始训练  共 {total_epochs} 轮")
    print("=" * 65 + "\n")

    # 外层进度条：轮次总览
    epoch_pbar = tqdm(
        range(1, total_epochs + 1),
        desc="总进度",
        ncols=100,
        position=0,
    )

    for epoch in epoch_pbar:
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            device, use_amp, epoch, total_epochs
        )
        val_loss = validate(
            model, val_loader,
            device, use_amp, epoch, total_epochs
        )
        scheduler.step(val_loss)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        # 外层进度条后缀
        epoch_pbar.set_postfix(
            train=f"{train_loss:.5f}",
            val=f"{val_loss:.5f}",
            lr=f"{current_lr:.1e}",
        )

        # 控制台单行摘要（在进度条下方打印，不干扰进度条）
        tqdm.write(
            f"Epoch [{epoch:4d}/{total_epochs}]  "
            f"Train: {train_loss:.6f}  "
            f"Val: {val_loss:.6f}  "
            f"LR: {current_lr:.2e}  "
            f"耗时: {elapsed:.1f}s"
        )

        # 写日志
        append_log(log_path, [epoch, f"{train_loss:.6f}",
                               f"{val_loss:.6f}", f"{elapsed:.1f}",
                               f"{current_lr:.2e}"])

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scaler, epoch, val_loss, best_path)
            tqdm.write(f"  ★ 最优模型已保存 (val_loss={val_loss:.6f})")

        # 每 N 轮定期保存
        if epoch % CONFIG['save_every'] == 0:
            ckpt_path = os.path.join(CONFIG['save_dir'], f'epoch_{epoch:04d}.pth')
            save_checkpoint(model, optimizer, scaler, epoch, val_loss, ckpt_path)
            tqdm.write(f"  → 定期检查点已保存: {ckpt_path}")

    print("\n\n训练完成！")
    print(f"最优验证损失 : {best_val_loss:.6f}")
    print(f"模型保存目录 : {CONFIG['save_dir']}")
    print(f"训练日志     : {log_path}")


if __name__ == "__main__":
    main()