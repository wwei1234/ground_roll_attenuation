# train_velocity.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeismicDataset
from src.utils import (
    ensure_dir, get_file_list, import_model_class, save_loss_csv, plot_loss_curve,
)
from src.losses import continuity_loss, velocity_range_loss

# ==================== 用户配置区（请在此处修改）====================
# 路径配置
DATA_DIR = r"D:\桌面\面波压制\深层数据面波压制\原始炮集"           # 原始地震数据文件夹（191炮）
LABEL_DIR = r"D:\桌面\面波压制\深层数据面波压制\masks"        # 标签文件夹（0-1标注）
MODEL_FILE = "U_Net_CBAM.py"     # 网络模型文件路径
MODEL_CLASS_NAME = "UNet"        # 目标网络类名，默认优先加载该类
SAVE_DIR = "./checkpoints_velocity"    # 模型保存文件夹（会自动创建）

# 测试集配置（指定炮号，文件名列表，如 ["shot_001.npy", "shot_050.npy"]）
TEST_FILES = ["real_data_gather183.npy"]  # 请在这里填入你要作为测试集的文件名

# 训练超参数
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0               # Windows 建议设为 0，避免多进程报错
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 验证集比例（剩余数据中验证集占的比例，9:1 即 val_ratio=0.1）
VAL_RATIO = 0.1
RANDOM_SEED = 42

# 物理约束损失权重（设为 0 可关闭对应约束）
LAMBDA_CONTINUITY = 0.1       # 连续性/平滑性约束权重
LAMBDA_VELOCITY = 0.05        # 速度范围约束权重

# 速度范围约束参数（根据实际地震采集参数修改）
DX = 25.0                     # 道间距，单位：米
DT = 0.004                    # 采样间隔，单位：秒
V_MIN = 200.0                 # 最小面波速度，单位：m/s
V_MAX = 1000.0                # 最大面波速度，单位：m/s

# 损失曲线保存路径
TRAIN_LOSS_CSV = "./train_loss_velocity.csv"
VAL_LOSS_CSV = "./val_loss_velocity.csv"
LOSS_CURVE_PNG = "./loss_curve_velocity.png"
# ================================================================



def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    lambda_cont, lambda_vel, dx, dt, v_min, v_max):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_cont = 0.0
    total_vel = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for data, label in pbar:
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # 1. 基础 BCE 损失（输入 logits，内部自动 sigmoid）
        loss_bce = criterion(output, label)
        
        # 2. 物理约束损失（基于概率图）
        pred_prob = torch.sigmoid(output)
        loss_cont = continuity_loss(pred_prob) if lambda_cont > 0 else torch.tensor(0.0, device=device)
        loss_vel = velocity_range_loss(pred_prob, dx, dt, v_min, v_max) if lambda_vel > 0 else torch.tensor(0.0, device=device)
        
        # 3. 总损失
        loss = loss_bce + lambda_cont * loss_cont + lambda_vel * loss_vel
        
        loss.backward()
        optimizer.step()
        
        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_bce += loss_bce.item() * batch_size
        total_cont += loss_cont.item() * batch_size
        total_vel += loss_vel.item() * batch_size
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'bce': f"{loss_bce.item():.4f}",
            'cont': f"{loss_cont.item():.4f}",
            'vel': f"{loss_vel.item():.4f}"
        })
    
    n_samples = len(dataloader.dataset)
    return {
        'total': total_loss / n_samples,
        'bce': total_bce / n_samples,
        'cont': total_cont / n_samples,
        'vel': total_vel / n_samples
    }


def validate(model, dataloader, criterion, device,
             lambda_cont, lambda_vel, dx, dt, v_min, v_max):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_cont = 0.0
    total_vel = 0.0
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            output = model(data)
            
            loss_bce = criterion(output, label)
            pred_prob = torch.sigmoid(output)
            loss_cont = continuity_loss(pred_prob) if lambda_cont > 0 else torch.tensor(0.0, device=device)
            loss_vel = velocity_range_loss(pred_prob, dx, dt, v_min, v_max) if lambda_vel > 0 else torch.tensor(0.0, device=device)
            
            loss = loss_bce + lambda_cont * loss_cont + lambda_vel * loss_vel
            
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_bce += loss_bce.item() * batch_size
            total_cont += loss_cont.item() * batch_size
            total_vel += loss_vel.item() * batch_size
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bce': f"{loss_bce.item():.4f}",
                'cont': f"{loss_cont.item():.4f}",
                'vel': f"{loss_vel.item():.4f}"
            })
    
    n_samples = len(dataloader.dataset)
    return {
        'total': total_loss / n_samples,
        'bce': total_bce / n_samples,
        'cont': total_cont / n_samples,
        'vel': total_vel / n_samples
    }


def main():
    # 创建保存目录
    ensure_dir(SAVE_DIR)
    
    # 动态导入模型
    model_class = import_model_class(MODEL_FILE, MODEL_CLASS_NAME)
    
    # 获取文件列表并划分数据集
    all_files = get_file_list(DATA_DIR, exclude_list=TEST_FILES)
    print(f"总样本数（排除测试集）: {len(all_files)}")
    print(f"测试集样本数: {len(TEST_FILES)}")
    
    if len(all_files) == 0:
        raise ValueError("没有可用的训练/验证样本，请检查路径和测试集配置")
    
    # 划分训练集和验证集（9:1）
    train_size = int(len(all_files) * (1 - VAL_RATIO))
    val_size = len(all_files) - train_size
    
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(all_files))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]
    
    print(f"训练集: {len(train_files)} 炮, 验证集: {len(val_files)} 炮")
    
    train_dataset = SeismicDataset(DATA_DIR, LABEL_DIR, train_files, transform=True)
    val_dataset = SeismicDataset(DATA_DIR, LABEL_DIR, val_files, transform=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    # 初始化模型
    model = model_class().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # 模型输出 raw logits，内部自动 sigmoid
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 训练记录（分项记录）
    train_losses = {'total': [], 'bce': [], 'cont': [], 'vel': []}
    val_losses = {'total': [], 'bce': [], 'cont': [], 'vel': []}
    best_val_loss = float('inf')
    
    print(f"\n开始训练，设备: {DEVICE}")
    print(f"物理约束: 连续性权重={LAMBDA_CONTINUITY}, 速度范围权重={LAMBDA_VELOCITY}")
    print(f"采集参数: dx={DX}m, dt={DT}s, 速度范围=[{V_MIN}, {V_MAX}] m/s")
    print("=" * 60)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            LAMBDA_CONTINUITY, LAMBDA_VELOCITY, DX, DT, V_MIN, V_MAX
        )
        val_result = validate(
            model, val_loader, criterion, DEVICE,
            LAMBDA_CONTINUITY, LAMBDA_VELOCITY, DX, DT, V_MIN, V_MAX
        )
        
        # 记录损失
        for key in train_losses.keys():
            train_losses[key].append(train_result[key])
            val_losses[key].append(val_result[key])
        
        print(f"Train -> Total: {train_result['total']:.6f} | BCE: {train_result['bce']:.6f} | "
              f"Cont: {train_result['cont']:.6f} | Vel: {train_result['vel']:.6f}")
        print(f"Val   -> Total: {val_result['total']:.6f} | BCE: {val_result['bce']:.6f} | "
              f"Cont: {val_result['cont']:.6f} | Vel: {val_result['vel']:.6f}")
        
        # 每 10 轮保存模型
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_result['total'],
                'val_loss': val_result['total'],
            }, checkpoint_path)
            print(f"模型已保存: {checkpoint_path}")
        
        # 保存最佳模型（以总验证损失为准）
        if val_result['total'] < best_val_loss:
            best_val_loss = val_result['total']
            best_path = os.path.join(SAVE_DIR, "model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_path)
    
    # 保存损失 CSV
    save_loss_csv(train_losses, TRAIN_LOSS_CSV)
    save_loss_csv(val_losses, VAL_LOSS_CSV)
    print(f"\n损失已保存: {TRAIN_LOSS_CSV}, {VAL_LOSS_CSV}")
    
    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, LOSS_CURVE_PNG)
    print(f"损失曲线已保存: {LOSS_CURVE_PNG}")
    
    print("=" * 60)
    print("训练完成！")


if __name__ == "__main__":
    main()