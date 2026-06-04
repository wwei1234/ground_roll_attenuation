import os
import csv
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from U_Net_CBAM import UNet
from dataset2 import ManualMaskSeismicDataset

# ============================================================
#  ★ 超参数配置
# ============================================================
CONFIG = {
    "epochs"         : 200,
    "batch_size"     : 4,
    "learning_rate"  : 1e-4,
    "val_ratio"      : 0.2,
    "num_workers"    : 0,
    "seed"           : 42,
    "use_amp"        : True,
    "save_dir"       : "CU-Net_3_9",
    "log_dir"        : "logs_3_9_CU-Net",
    "save_every"     : 10,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",

    # 频率约束损失权重（0 则退化为纯重建损失）
    "lambda_freq"    : 0,
    # 面波主频上限（Hz），根据你的工区调整
    "freq_cutoff_hz" : 8.0,
    # 时间采样间隔（秒），根据你的数据修改
    "dt"             : 0.004,
}

# ============================================================
#  ★ Dataset 配置（只需一组）
# ============================================================
DATASET_CONFIG = dict(
    gather_dir      = r"原始炮集",
    mask_dir        = r"masks",
    fan_mask_dir    = r"fan_masks",
    total           = 192,
    prefix_gather   = "real_data_gather",
    prefix_mask     = "mask_",
    prefix_fan_mask = "fan_mask_",
    max_time_samples    = None,
    mask_k              = 0.5,
    enable_augmentation = True,
    loss_on_full_non_sw = True,
    normalize           = True,
)

# ============================================================
#  损失函数
# ============================================================

def masked_l1_loss(pred, target, mask):
    """仅在 mask=1 的区域计算 L1 损失。"""
    diff  = torch.abs(pred - target)
    denom = mask.sum().clamp(min=1.0)
    return (diff * mask).sum() / denom


def freq_constraint_loss(pred, sw_mask, dt=0.002, cutoff_hz=20.0):
    """
    面波区域低频压制约束。

    对网络在面波区域的预测做 FFT，惩罚低于 cutoff_hz 的频率能量。
    不需要 ground truth，仅依赖物理先验。

    Parameters
    ----------
    pred     : (B, 1, T, X)  网络输出
    sw_mask  : (B, 1, T, X)  面波区域 mask（1=面波）
    dt       : 时间采样间隔（秒）
    cutoff_hz: 低频截止频率（Hz）
    """
    if sw_mask.sum() < 1:
        return torch.tensor(0.0, device=pred.device)

    B, C, T, X = pred.shape
    sw_pred = pred * sw_mask                               # 只取面波区域

    spec  = torch.fft.rfft(sw_pred, dim=2)                # (B,1,T//2+1,X)
    power = spec.real ** 2 + spec.imag ** 2

    freqs = torch.fft.rfftfreq(T, d=dt).to(pred.device)   # (T//2+1,)
    low_freq_mask = (freqs < cutoff_hz).float()[None, None, :, None]

    low_energy   = (power * low_freq_mask).sum()
    total_energy = power.sum().clamp(min=1e-8)

    return low_energy / total_energy

# ============================================================
#  工具函数
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """list of dict → dict of tensor，添加 channel 维度。"""
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if isinstance(vals[0], np.ndarray):
            t = torch.from_numpy(np.stack(vals, axis=0))
            if k in ('input_data', 'label_data', 'original_data'):
                t = t.unsqueeze(1)
            elif k in ('loss_region', 'surface_wave_mask',
                       'fan_mask', 'non_sw_region'):
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


def init_log(log_path):
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'train_recon', 'train_freq',
            'val_loss',   'val_recon',   'val_freq',
            'epoch_time_s', 'lr'
        ])


def append_log(log_path, row):
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

# ============================================================
#  训练 / 验证 单轮
# ============================================================

def run_epoch(model, loader, optimizer, scaler,
              device, use_amp, epoch, total_epochs,
              lambda_freq, freq_cutoff_hz, dt,
              is_train: bool):

    model.train() if is_train else model.eval()
    total_loss = total_recon = total_freq = 0.0
    tag = "训练" if is_train else "验证"

    pbar = tqdm(loader, desc=f"Epoch [{epoch:4d}/{total_epochs}] {tag}",
                ncols=110, leave=False)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(pbar, 1):
            inp     = batch['input_data'].to(device)
            label   = batch['label_data'].to(device)
            mask    = batch['loss_region'].to(device)
            sw_mask = batch['surface_wave_mask'].to(device)

            if is_train:
                optimizer.zero_grad()

            with autocast(enabled=use_amp):
                pred       = model(inp)
                loss_recon = masked_l1_loss(pred, label, mask)
                loss_freq  = freq_constraint_loss(pred, sw_mask,
                                                  dt=dt, cutoff_hz=freq_cutoff_hz)
                loss       = loss_recon + lambda_freq * loss_freq

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss  += loss.item()
            total_recon += loss_recon.item()
            total_freq  += loss_freq.item()
            pbar.set_postfix(
                loss=f"{total_loss/step:.5f}",
                recon=f"{total_recon/step:.5f}",
                freq=f"{total_freq/step:.5f}",
            )

    n = len(loader)
    return total_loss / n, total_recon / n, total_freq / n

# ============================================================
#  主训练流程
# ============================================================

def main():
    set_seed(CONFIG['seed'])
    device  = torch.device(CONFIG['device'])
    use_amp = CONFIG['use_amp'] and (CONFIG['device'] == 'cuda')

    lf  = CONFIG['lambda_freq']
    fhz = CONFIG['freq_cutoff_hz']
    dt  = CONFIG['dt']

    print(f"\n使用设备     : {device}")
    print(f"混合精度     : {'开启' if use_amp else '关闭'}")
    print(f"频率约束     : lambda={lf}  cutoff={fhz}Hz  dt={dt}s")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'],  exist_ok=True)
    log_path  = os.path.join(CONFIG['log_dir'],  'train_log.csv')
    best_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
    init_log(log_path)

    # ---------- 数据集 ----------
    print("\n正在构建数据集...")
    dataset  = ManualMaskSeismicDataset(**DATASET_CONFIG)
    total_len = len(dataset)
    val_len   = int(total_len * CONFIG['val_ratio'])
    train_len = total_len - val_len

    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    print(f"训练集: {train_len}  |  验证集: {val_len}")

    loader_kwargs = dict(
        batch_size  = CONFIG['batch_size'],
        num_workers = CONFIG['num_workers'],
        collate_fn  = collate_fn,
        pin_memory  = (CONFIG['device'] == 'cuda'),
    )
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kwargs)

    # ---------- 模型 ----------
    model     = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    total_epochs  = CONFIG['epochs']

    print("\n" + "=" * 65)
    print(f"开始训练  共 {total_epochs} 轮")
    print("=" * 65 + "\n")

    epoch_pbar = tqdm(range(1, total_epochs + 1), desc="总进度",
                      ncols=110, position=0)

    for epoch in epoch_pbar:
        t0 = time.time()

        tr_loss, tr_recon, tr_freq = run_epoch(
            model, train_loader, optimizer, scaler,
            device, use_amp, epoch, total_epochs,
            lf, fhz, dt, is_train=True
        )
        va_loss, va_recon, va_freq = run_epoch(
            model, val_loader, optimizer, scaler,
            device, use_amp, epoch, total_epochs,
            lf, fhz, dt, is_train=False
        )
        scheduler.step(va_loss)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        epoch_pbar.set_postfix(
            train=f"{tr_loss:.5f}", val=f"{va_loss:.5f}",
            lr=f"{current_lr:.1e}"
        )
        tqdm.write(
            f"Epoch [{epoch:4d}/{total_epochs}]  "
            f"Train: {tr_loss:.6f} (recon={tr_recon:.6f} freq={tr_freq:.6f})  "
            f"Val: {va_loss:.6f} (recon={va_recon:.6f} freq={va_freq:.6f})  "
            f"LR: {current_lr:.2e}  耗时: {elapsed:.1f}s"
        )
        append_log(log_path, [
            epoch,
            f"{tr_loss:.6f}", f"{tr_recon:.6f}", f"{tr_freq:.6f}",
            f"{va_loss:.6f}", f"{va_recon:.6f}", f"{va_freq:.6f}",
            f"{elapsed:.1f}", f"{current_lr:.2e}",
        ])

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            save_checkpoint(model, optimizer, scaler, epoch, va_loss, best_path)
            tqdm.write(f"  ★ 最优模型已保存 (val_loss={va_loss:.6f})")

        if epoch % CONFIG['save_every'] == 0:
            ckpt = os.path.join(CONFIG['save_dir'], f'epoch_{epoch:04d}.pth')
            save_checkpoint(model, optimizer, scaler, epoch, va_loss, ckpt)
            tqdm.write(f"  → 定期检查点: {ckpt}")

    print(f"\n训练完成！最优验证损失: {best_val_loss:.6f}")
    print(f"模型目录: {CONFIG['save_dir']}  日志: {log_path}")


if __name__ == "__main__":
    main()