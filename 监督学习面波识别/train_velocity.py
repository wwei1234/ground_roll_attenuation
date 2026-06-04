# train.py
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def continuity_loss(pred_prob, epsilon=1e-3):
    """
    连续性/平滑性约束（Spatial Continuity）
    基于 Charbonnier TV 损失，鼓励预测的面波区域为连续条带，
    惩罚不连续性、孤立点和锯齿状边缘。
    
    pred_prob: 模型输出的概率图 (B, C, T, X)，已 sigmoid
    """
    # 时间方向梯度
    grad_t = pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]
    # 空间方向（道方向）梯度
    grad_x = pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]
    
    # 对齐维度
    grad_t = grad_t[:, :, :, :-1]
    grad_x = grad_x[:, :, :-1, :]
    
    # Charbonnier 惩罚：sqrt(x^2 + epsilon^2)，比 L1/L2 更适合保持边缘
    loss_t = torch.mean(torch.sqrt(grad_t ** 2 + epsilon ** 2))
    loss_x = torch.mean(torch.sqrt(grad_x ** 2 + epsilon ** 2))
    
    return loss_t + loss_x


def velocity_range_loss(pred_prob, dx, dt, v_min, v_max, epsilon=1e-6):
    """
    速度范围约束损失（Velocity Range Constraint）
    约束预测面波区域的局部同相轴斜率对应的速度在 [v_min, v_max] 范围内。
    
    物理原理：
        在炮集记录 (t-x) 中，面波同相轴斜率 slope = dt_pixel / dx_pixel，
        对应物理速度 v = dx / (slope * dt)。
        因此面波在图像上的合理斜率范围为：
            slope_min = dx / (v_max * dt)   （高速面波，斜率小）
            slope_max = dx / (v_min * dt)   （低速面波，斜率大）
    
    pred_prob: 模型输出的概率图 (B, C, T, X)，已 sigmoid
    dx: 道间距（米）
    dt: 采样间隔（秒）
    v_min, v_max: 面波合理速度范围（m/s）
    """
    # 对概率图做轻微平滑，使梯度更稳定
    pred_smooth = F.avg_pool2d(pred_prob, kernel_size=3, stride=1, padding=1)
    
    # 计算平滑后的梯度
    g_t = pred_smooth[:, :, 1:, :] - pred_smooth[:, :, :-1, :]  # 时间方向
    g_x = pred_smooth[:, :, :, 1:] - pred_smooth[:, :, :, :-1]  # 道方向
    
    # 对齐维度 -> (B, C, T-1, X-1)
    g_t = g_t[:, :, :, :-1]
    g_x = g_x[:, :, :-1, :]
    
    # 局部斜率（像素/像素）：|dt/dx|
    slope = torch.abs(g_t) / (torch.abs(g_x) + epsilon)
    
    # 速度范围对应的斜率边界
    # v = dx / (slope * dt)  =>  slope = dx / (v * dt)
    slope_min = dx / (v_max * dt)   # 高速对应小斜率
    slope_max = dx / (v_min * dt)   # 低速对应大斜率
    
    # 只在预测为面波的区域（概率 > 0.5）且 x 方向梯度显著处进行约束
    # 避免除以 0 和背景区域的干扰
    valid_mask = (pred_prob[:, :, :-1, :-1] > 0.5) & (torch.abs(g_x) > 0.01)
    
    # 惩罚：
    #   slope < slope_min：速度太高（像水平层位），不像面波
    #   slope > slope_max：速度太低（像陡倾噪声），不像面波
    penalty_low = F.relu(slope_min - slope)   # 速度过高惩罚
    penalty_high = F.relu(slope - slope_max)  # 速度过低惩罚
    
    penalty = penalty_low + penalty_high
    
    if valid_mask.sum() > 0:
        loss = (penalty * valid_mask.float()).sum() / valid_mask.sum()
    else:
        loss = torch.tensor(0.0, device=pred_prob.device)
    
    return loss


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


def save_loss_csv(loss_dict, csv_path):
    """保存各分项损失到 CSV"""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'total_loss', 'bce_loss', 'continuity_loss', 'velocity_loss'])
        n_epochs = len(loss_dict['total'])
        for i in range(n_epochs):
            writer.writerow([
                i + 1,
                loss_dict['total'][i],
                loss_dict['bce'][i],
                loss_dict['cont'][i],
                loss_dict['vel'][i]
            ])


def plot_loss_curve(train_losses, val_losses, save_path):
    """绘制并保存损失曲线（包含总损失和各分项）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(train_losses['total']) + 1)
    
    # 总损失
    axes[0, 0].plot(epochs, train_losses['total'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_losses['total'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # BCE 损失
    axes[0, 1].plot(epochs, train_losses['bce'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, val_losses['bce'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_title('BCE Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 连续性损失
    axes[1, 0].plot(epochs, train_losses['cont'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, val_losses['cont'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title('Continuity Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 速度范围损失
    axes[1, 1].plot(epochs, train_losses['vel'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, val_losses['vel'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_title('Velocity Range Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    # 创建保存目录
    ensure_dir(SAVE_DIR)
    
    # 动态导入模型（从 model.py 中导入网络类）
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", MODEL_FILE)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    # 自动获取网络类
    model_class = None
    for attr_name in dir(model_module):
        attr = getattr(model_module, attr_name)
        if attr_name == MODEL_CLASS_NAME and isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
            model_class = attr
            break

    if model_class is None:
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