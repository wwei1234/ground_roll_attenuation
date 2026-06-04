# train.py
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== 用户配置区（请在此处修改）====================
# 路径配置
DATA_DIR = r"D:\桌面\面波压制\深层数据面波压制\原始炮集"           # 原始地震数据文件夹（191炮）
LABEL_DIR = r"D:\桌面\面波压制\深层数据面波压制\masks"        # 标签文件夹（0-1标注）
MODEL_FILE = "U_Net_CBAM.py"     # 网络模型文件路径
MODEL_CLASS_NAME = "UNet"        # 目标网络类名，默认优先加载该类
SAVE_DIR = "./checkpoints_6_3"       # 模型保存文件夹（会自动创建）
LOGS_DIR = "./logs_6_3"              # 训练日志保存文件夹（会自动创建）

# 测试集配置（指定炮号，文件名列表，如 ["shot_001.npy", "shot_050.npy"]）
TEST_FILES = ["real_data_gather183.npy", "real_data_gather182.npy", "real_data_gather184.npy", "real_data_gather185.npy"]  # 请在这里填入你要作为测试集的文件名

# 训练超参数
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 验证集比例（剩余数据中验证集占的比例，9:1 即 val_ratio=0.1）
VAL_RATIO = 0.1
RANDOM_SEED = 42

# 损失曲线保存路径
TRAIN_LOSS_CSV = "train_loss.csv"
VAL_LOSS_CSV = "val_loss.csv"
LOSS_CURVE_PNG = "loss_curve.png"
# ================================================================


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_file_list(data_dir, exclude_list=None):
    """获取文件夹中所有 .npy 文件，排除测试集"""
    if exclude_list is None:
        exclude_list = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    files = [f for f in files if f not in exclude_list]
    return files


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for data, label in pbar:
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss.item() * data.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    
    return total_loss / len(dataloader.dataset)


def save_loss_csv(loss_list, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i, loss in enumerate(loss_list, 1):
            writer.writerow([i, loss])


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    # 创建保存目录
    ensure_dir(SAVE_DIR)
    ensure_dir(LOGS_DIR)
    
    # 接合日志路径
    train_loss_csv = os.path.join(LOGS_DIR, TRAIN_LOSS_CSV)
    val_loss_csv = os.path.join(LOGS_DIR, VAL_LOSS_CSV)
    loss_curve_png = os.path.join(LOGS_DIR, LOSS_CURVE_PNG)
    
    # 动态导入模型（从 model.py 中导入网络类）
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", MODEL_FILE)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    # 自动获取网络类
    model_class = None
    # 优先加载配置的目标模型类名
    for attr_name in dir(model_module):
        attr = getattr(model_module, attr_name)
        if attr_name == MODEL_CLASS_NAME and isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
            model_class = attr
            break

    if model_class is None:
        # 如果未找到指定类，回退到第一个 nn.Module 子类
        for attr_name in dir(model_module):
            attr = getattr(model_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                model_class = attr
                print(f"检测到网络模型类: {attr_name}")
                break

    if model_class is None:
        raise RuntimeError("未能在模型文件中找到继承自 nn.Module 的类，请检查模型文件")
    
    # 获取文件列表并划分数据集
    all_files = get_file_list(DATA_DIR, exclude_list=TEST_FILES)
    print(f"总样本数（排除测试集）: {len(all_files)}")
    print(f"测试集样本数: {len(TEST_FILES)}")
    
    if len(all_files) == 0:
        raise ValueError("没有可用的训练/验证样本，请检查路径和测试集配置")
    
    # 划分训练集和验证集（9:1）
    train_size = int(len(all_files) * (1 - VAL_RATIO))
    val_size = len(all_files) - train_size
    
    # 为了可复现，先设置随机种子
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(all_files))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]
    
    print(f"训练集: {len(train_files)} 炮, 验证集: {len(val_files)} 炮")
    
    # 导入 Dataset
    from dataset import SeismicDataset
    
    train_dataset = SeismicDataset(DATA_DIR, LABEL_DIR, train_files, transform=True)
    val_dataset = SeismicDataset(DATA_DIR, LABEL_DIR, val_files, transform=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 初始化模型
    model = model_class().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # 模型输出是 raw logits，因此使用 BCEWithLogitsLoss 更稳定
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n开始训练，设备: {DEVICE}")
    print("=" * 60)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 每 10 轮保存模型
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"模型已保存: {checkpoint_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(SAVE_DIR, "model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_path)
    
    # 保存损失 CSV
    save_loss_csv(train_losses, train_loss_csv)
    save_loss_csv(val_losses, val_loss_csv)
    print(f"\n损失已保存: {train_loss_csv}, {val_loss_csv}")
    
    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, loss_curve_png)
    print(f"损失曲线已保存: {loss_curve_png}")
    
    print("=" * 60)
    print("训练完成！")


if __name__ == "__main__":
    main()