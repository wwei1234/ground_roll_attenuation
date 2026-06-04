import torch
import os
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np 
from dataset import SeismicMaskDataset  
from field_data import SeismicMaskDataset_FIELD  
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from U_Net_CBAM import UNet 

def masked_mae_loss(pred, target, mask):
    """
    只在掩码位置计算MAE损失
    
    Args:
        pred: 预测结果 [B, 1, H, W] 或 [B, H, W]
        target: 目标标签 [B, H, W]
        mask: 掩码位置 [B, H, W], bool类型
    
    Returns:
        loss: 掩码位置的平均绝对误差
    """
    # 确保维度一致
    if pred.dim() == 4:  # [B, 1, H, W]
        pred = pred.squeeze(1)  # [B, H, W]
    
    # 只计算掩码位置的损失
    masked_pred = pred[mask]
    masked_target = target[mask]
    
    # 计算MAE
    loss = torch.mean(torch.abs(masked_pred - masked_target))
    
    return loss

# 训练参数
batch_size = 32
num_epochs = 500
learning_rate = 4e-4
save_interval = 100  # 每100轮保存一次模型

# 模型保存路径
model_dir = r"D:\桌面\面波压制\model\model_field_1112"
log_dir = r"D:\桌面\面波压制\train_log\training_logs_field_1112"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# # ========== 配置多个数据集参数 ==========
# dataset_SYN_configs = [
#     {
#         'name': 'Dataset_1',
#         'segy_path': r"D:\桌面\面波压制\data\dataset1_260.sgy",
#         'shot_center': 24,
#         'dt': 0.00025,
#         'shotnum': 260,
#         'max_time_samples': 1200,
#         # 面波识别参数
#         'window_width': 0.012,
#         'time_shift': 0.002,
#         'sw_threshold': 0.08,
#         # 直达波去除参数
#         'remove_direct': True,
#         'direct_velocity': 800,
#         'mute_time0': 0.010,
#         'dx': 2,
#         'taper': 20,
#         "non_sw_density_range" : (0, 0.03) , # 非面波区域活跃像素密度
#         "mask_k" : 0.5  ,# 面波和非面波统一使用 [-k, k] 高斯分布
#         "enable_augmentation" : False  # 数据增强
#     },
#     {
#         'name': 'Dataset_2',
#         'segy_path': r"D:\桌面\面波压制\data\dataset2_260.sgy",
#         'shot_center': 24,
#         'dt': 0.00025,
#         'shotnum': 260,
#         'max_time_samples': 1200,
#         # 面波识别参数
#         'window_width': 0.012,
#         'time_shift': 0.002,
#         'sw_threshold': 0.08,
#         # 直达波去除参数
#         'remove_direct': True,
#         'direct_velocity': 800,
#         'mute_time0': 0.010,
#         'dx': 2,
#         'taper': 20,
#         "non_sw_density_range" : (0, 0.08) , # 非面波区域活跃像素密度
#         "mask_k" : 0.5  ,# 面波和非面波统一使用 [-k, k] 高斯分布
#         "enable_augmentation" : False  # 数据增强
#     },
#     {
#         'name': 'Dataset_3',
#         'segy_path': r"D:\桌面\面波压制\data\dataset3_260.sgy",
#         'shot_center': 24,
#         'dt': 0.00025,
#         'shotnum': 260,
#         'max_time_samples': 1200,
#         # 面波识别参数
#         'window_width': 0.012,
#         'time_shift': 0.002,
#         'sw_threshold': 0.08,
#         # 直达波去除参数
#         'remove_direct': True,
#         'direct_velocity': 800,
#         'mute_time0': 0.010,
#         'dx': 2,
#         'taper': 20,
#         "non_sw_density_range" : (0, 0.1) , # 非面波区域活跃像素密度
#         "mask_k" : 0.5  ,# 面波和非面波统一使用 [-k, k] 高斯分布
#         "enable_augmentation" : False  # 数据增强
#     },
#     {
#         'name': 'Dataset_4',
#         'segy_path': r"D:\桌面\面波压制\data\dataset4_700.sgy",
#         'shot_center': 24,
#         'dt': 0.00025,
#         'shotnum': 700,
#         'max_time_samples': 1200,
#         # 面波识别参数
#         'window_width': 0.012,
#         'time_shift': 0.002,
#         'sw_threshold': 0.08,
#         # 直达波去除参数
#         'remove_direct': True,
#         'direct_velocity': 800,
#         'mute_time0': 0.010,
#         'dx': 2,
#         'taper': 20,
#         "non_sw_density_range" : (0, 0.05) , # 非面波区域活跃像素密度
#         "mask_k" : 0.5  ,# 面波和非面波统一使用 [-k, k] 高斯分布
#         "enable_augmentation" : False  # 数据增强
#     },
# ]

# datasets_SYN = []
# for i, config in enumerate(dataset_SYN_configs):
#     print(f"\n创建数据集 {i+1}/{len(dataset_SYN_configs)}: {config['name']}")
#     print("-" * 70)
    
#     dataset = SeismicMaskDataset(
#         segy_path=config['segy_path'],
#         shot_center=config['shot_center'],
#         dt=config['dt'],
#         shotnum=config['shotnum'],
#         max_time_samples=config['max_time_samples'],
#         # 面波识别参数
#         window_width=config['window_width'],
#         time_shift=config['time_shift'],
#         sw_threshold=config['sw_threshold'],
#         # 直达波去除参数
#         remove_direct=config['remove_direct'],
#         direct_velocity=config['direct_velocity'],
#         mute_time0=config['mute_time0'],
#         dx=config['dx'],
#         taper=config['taper'],
#         # 通用参数
#         non_sw_density_range=config["non_sw_density_range"],
#         enable_augmentation=config["enable_augmentation"],
#         mask_k=config["mask_k"]
#     )
#     datasets_SYN.append(dataset)
#     print(f"✓ {config['name']} 创建完成: {len(dataset)} 个样本")

dataset_FIELD_configs = [
    {
        'name': 'Dataset_1',
        'segy_path': r"D:\桌面\面波压制\data\modified_SN03.sgy",
        'shot_center_mode':'cyclic',  # 'fixed': 固定炮点, 'cyclic': 循环炮点
        'shot_center_fixed':24,      # 固定模式的炮点位置
        'shot_center_start':12,      # 循环模式的起始炮点
        'shot_center_end':35,
        'dt': 0.00025,
        'shotnum': 168,
        'max_time_samples': 1200,
        # 面波识别参数
        'window_width': 0.012,
        'time_shift': 0.002,
        'sw_threshold': 0.08,
        # 直达波去除参数
        'remove_direct': True,
        'direct_velocity': 1200,
        'mute_time0': 0.010,
        'dx': 2,
        'taper': 20,
        "non_sw_density_range" : (0, 0.03) , # 非面波区域活跃像素密度
        "mask_k" : 0.5  ,# 面波和非面波统一使用 [-k, k] 高斯分布
        "enable_augmentation" : False  # 数据增强
    },
    {
        'name': 'Dataset_2',
        'segy_path': r"D:\桌面\面波压制\data\modified_SN04.sgy",
        'shot_center_mode':'cyclic',  # 'fixed': 固定炮点, 'cyclic': 循环炮点
        'shot_center_fixed':24,      # 固定模式的炮点位置
        'shot_center_start':12,      # 循环模式的起始炮点
        'shot_center_end':35,
        'dt': 0.00025,
        'shotnum': 168,
        'max_time_samples': 1200,
        # 面波识别参数
        'window_width': 0.012,
        'time_shift': 0.002,
        'sw_threshold': 0.08,
        # 直达波去除参数
        'remove_direct': True,
        'direct_velocity': 1200,
        'mute_time0': 0.010,
        'dx': 2,
        'taper': 20,
        "non_sw_density_range" : (0, 0.08) , # 非面波区域活跃像素密度
        "mask_k" : 0.5  ,# 面波和非面波统一使用 [-k, k] 高斯分布
        "enable_augmentation" : False  # 数据增强
    },
    {
        'name': 'Dataset_3',
        'segy_path': r"D:\桌面\面波压制\data\modified_SN05.sgy",
        'shot_center_mode':'cyclic',  # 'fixed': 固定炮点, 'cyclic': 循环炮点
        'shot_center_fixed':24,      # 固定模式的炮点位置
        'shot_center_start':12,      # 循环模式的起始炮点
        'shot_center_end':35,
        'dt': 0.00025,
        'shotnum': 168,
        'max_time_samples': 1200,
        # 面波识别参数
        'window_width': 0.012,
        'time_shift': 0.002,
        'sw_threshold': 0.08,
        # 直达波去除参数
        'remove_direct': True,
        'direct_velocity': 1200,
        'mute_time0': 0.010,
        'dx': 2,
        'taper': 20,
        "non_sw_density_range" : (0, 0.1) , # 非面波区域活跃像素密度
        "mask_k" : 0.5  ,# 面波和非面波统一使用 [-k, k] 高斯分布
        "enable_augmentation" : False  # 数据增强
    },
]

datasets_FIELD = []
for i, config in enumerate(dataset_FIELD_configs):
    print(f"\n创建数据集 {i+1}/{len(dataset_FIELD_configs)}: {config['name']}")
    print("-" * 70)
    
    dataset = SeismicMaskDataset_FIELD(
        segy_path=config['segy_path'],
        shot_center_mode=config['shot_center_mode'],
        shot_center_fixed=config['shot_center_fixed'],
        shot_center_start=config['shot_center_start'],
        shot_center_end=config['shot_center_end'],
        dt=config['dt'],
        shotnum=config['shotnum'],
        max_time_samples=config['max_time_samples'],
        # 面波识别参数
        window_width=config['window_width'],
        time_shift=config['time_shift'],
        sw_threshold=config['sw_threshold'],
        # 直达波去除参数
        remove_direct=config['remove_direct'],
        direct_velocity=config['direct_velocity'],
        mute_time0=config['mute_time0'],
        dx=config['dx'],
        taper=config['taper'],
        # 通用参数
        non_sw_density_range=config["non_sw_density_range"],
        enable_augmentation=config["enable_augmentation"],
        mask_k=config["mask_k"]
    )
    
    datasets_FIELD.append(dataset)
    print(f"✓ {config['name']} 创建完成: {len(dataset)} 个样本")

# 合并所有数据集
print("\n" + "=" * 70)
print("合并所有数据集...")
print("=" * 70)
combined_dataset = ConcatDataset(datasets_FIELD)

print(f"\n合并后的数据集信息:")
print(f"  总数据集数量: {len(datasets_FIELD)}")

# 创建数据加载器
dataloader = DataLoader(
    combined_dataset, 
    batch_size=batch_size, 
    shuffle=True,  # 重要：打乱所有数据集的样本
    num_workers=0
)

print(f"  批次数: {len(dataloader)}")
print("=" * 70)

# ========== 初始化模型 ==========
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"\n使用设备: {device}")

model = UNet(in_channels=1, num_classes=1, base_c=64)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = GradScaler()

# 记录训练损失
train_losses = []
# ========== 训练循环 ==========
print("\n" + "=" * 70)
print("开始训练（统一高斯掩码策略 - 多数据集）...")
print("  策略：输入面波和非面波掩码均为高斯分布")
print("  标签：面波区域与输入相同，非面波区域为原始数据")
print("  损失：仅在非面波区域的掩码位置计算")
print("=" * 70)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # 使用tqdm显示进度条
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
    
    for batch_idx, batch in enumerate(pbar):
        # 获取数据（统一高斯掩码策略）
        input_data = batch['input_data'].unsqueeze(1).float().to(device)  # [B, 1, H, W] 网络输入
        label_data = batch['label_data'].float().to(device)               # [B, H, W] 网络标签
        non_sw_mask_positions = batch['non_sw_mask_positions'].bool().to(device)  # [B, H, W] 损失计算位置
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast(device_type=device_type):
            outputs = model(input_data)  # [B, 1, H, W]
            
            # 只在非面波区域的掩码位置计算损失
            loss = masked_mae_loss(outputs, label_data, non_sw_mask_positions)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        # 更新进度条显示
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # 计算平均损失
    avg_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # 每save_interval轮保存一次模型
    if (epoch + 1) % save_interval == 0:
        model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  → 模型已保存: {model_path}")


# ========== 保存最终模型 ==========
final_model_path = os.path.join(model_dir, "model_final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"\n最终模型已保存: {final_model_path}")


# ========== 保存训练损失 ==========
loss_array = np.array(train_losses)
loss_file = os.path.join(log_dir, 'train_losses.npy')
np.save(loss_file, loss_array)
print(f"训练损失已保存: {loss_file}")

# ========== 绘制损失曲线 ==========
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss (MAE)', color='blue')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MAE)', fontsize=12)
plt.title('Training Loss Curve (Unified Gaussian - Multi Dataset)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_curve_path = os.path.join(log_dir, 'loss_curve.png')
plt.savefig(loss_curve_path, dpi=300)
print(f"损失曲线已保存: {loss_curve_path}")
plt.close()


