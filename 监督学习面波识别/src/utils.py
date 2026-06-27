"""
公共工具函数
从 train.py / train_velocity.py / predict.py / dataset.py 中提取的重复代码，
逻辑完全保持不变。
"""
import os
import re
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def ensure_dir(dir_path):
    """如果目录不存在则创建。"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_file_list(data_dir, exclude_list=None):
    """获取文件夹中所有 .npy 文件，排除指定的测试集文件。"""
    if exclude_list is None:
        exclude_list = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    files = [f for f in files if f not in exclude_list]
    return files


def map_label_filename(data_filename):
    """根据数据文件名生成标签文件名。
    默认规则：如果标签文件名与数据文件名不同，则尝试从数据名中提取数字，
    并生成 mask_###.npy 格式。
    """
    base = os.path.splitext(data_filename)[0]
    digits = re.findall(r"\d+", base)
    if digits:
        num = digits[-1]
        return f"mask_{int(num):03d}.npy"
    return data_filename


def import_model_class(model_file, model_class_name):
    """从指定的模型文件中动态导入模型类。

    Args:
        model_file: 模型 .py 文件路径
        model_class_name: 目标类名（如 "UNet"）

    Returns:
        nn.Module 子类

    Raises:
        RuntimeError: 未找到匹配的类时抛出
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model_class = None
    # 优先加载指定类名的类
    for attr_name in dir(model_module):
        attr = getattr(model_module, attr_name)
        if (attr_name == model_class_name
                and isinstance(attr, type)
                and issubclass(attr, nn.Module)
                and attr != nn.Module):
            model_class = attr
            break

    if model_class is None:
        # 回退到第一个 nn.Module 子类
        for attr_name in dir(model_module):
            attr = getattr(model_module, attr_name)
            if (isinstance(attr, type)
                    and issubclass(attr, nn.Module)
                    and attr != nn.Module):
                model_class = attr
                print(f"检测到网络模型类: {attr_name}")
                break

    if model_class is None:
        raise RuntimeError("未能在模型文件中找到继承自 nn.Module 的类，请检查模型文件")

    return model_class


def load_model(model_path, model_file, model_class_name, device):
    """加载模型并恢复权重。

    Args:
        model_path: .pth checkpoint 路径
        model_file: 模型 .py 文件路径
        model_class_name: 目标类名（如 "UNet"）
        device: torch device

    Returns:
        加载好权重的 model（eval 模式）
    """
    model_class = import_model_class(model_file, model_class_name)
    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"成功加载模型: {model_path} (Epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def save_loss_csv(loss_data, csv_path):
    """保存训练/验证损失到 CSV 文件。

    支持两种输入格式：
    1. 简单列表：loss_data 为 list of float，输出列为 ['epoch', 'loss']
    2. 字典：loss_data 为 dict of list，如
       {'total': [...], 'bce': [...], 'cont': [...], 'vel': [...]}
       输出列为 ['epoch', 'total_loss', 'bce_loss', 'continuity_loss', 'velocity_loss']
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if isinstance(loss_data, dict):
            # 多列格式（用于 train_velocity.py）
            writer.writerow(['epoch', 'total_loss', 'bce_loss', 'continuity_loss', 'velocity_loss'])
            n_epochs = len(loss_data['total'])
            for i in range(n_epochs):
                writer.writerow([
                    i + 1,
                    loss_data['total'][i],
                    loss_data['bce'][i],
                    loss_data['cont'][i],
                    loss_data['vel'][i],
                ])
        else:
            # 简单格式（用于 train.py）
            writer.writerow(['epoch', 'loss'])
            for i, loss in enumerate(loss_data, 1):
                writer.writerow([i, loss])


def plot_loss_curve(train_losses, val_losses, save_path):
    """绘制训练/验证损失曲线并保存。

    支持两种输入格式：
    1. 简单列表：单张图显示 Total Loss 曲线
    2. 字典：2x2 子图分别显示总损失、BCE、连续性、速度范围损失
    """
    if isinstance(train_losses, dict):
        # 多分量损失图（用于 train_velocity.py）
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
    else:
        # 简单损失图（用于 train.py）
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()