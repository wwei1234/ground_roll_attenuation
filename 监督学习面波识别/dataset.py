# Dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from src.utils import map_label_filename


class SeismicDataset(Dataset):
    """
    地震面波数据集
    数据格式要求：
        - 数据和标签分别存放在两个文件夹中
        - 文件名需一一对应（如 shot_001.npy 对应 shot_001.npy）
        - 数据形状建议: (时间采样点, 道数) 或 (道数, 时间采样点)
        - 标签为 0-1 二值矩阵，形状与数据一致
    """
    def __init__(self, data_dir, label_dir, file_list, transform=True):
        """
        Args:
            data_dir: 地震数据文件夹路径
            label_dir: 标签文件夹路径
            file_list: 该数据集包含的文件名列表
            transform: 是否进行数据增强（左右翻转）
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # 加载数据
        data_path = os.path.join(self.data_dir, filename)
        label_path = os.path.join(self.label_dir, filename)
        if not os.path.exists(label_path):
            alt_label_name = map_label_filename(filename)
            alt_label_path = os.path.join(self.label_dir, alt_label_name)
            if os.path.exists(alt_label_path):
                label_path = alt_label_path
            else:
                raise FileNotFoundError(
                    f"标签文件不存在: {label_path}，也未找到备选文件: {alt_label_path}"
                )
        
        data = np.load(data_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)
        
        # 数据形状统一处理：统一转为 (时间, 道数) -> (1, 时间, 道数)
        # 如果你的数据本身就是 (道数, 时间)，请注释掉下面这行或调整
        if data.ndim == 2:
            # 假设输入为 (time, trace)，添加通道维度
            data = np.expand_dims(data, axis=0)  # (1, time, trace)
            label = np.expand_dims(label, axis=0)  # (1, time, trace)
        elif data.ndim == 3:
            # 如果已经是 (channel, time, trace)，保持不变
            pass
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}，文件: {filename}")
        
        # 随机左右翻转（沿道数方向翻转，即空间方向）
        if self.transform and random.random() > 0.5:
            data = np.flip(data, axis=-1).copy()  # 翻转最后一个维度（道数）
            label = np.flip(label, axis=-1).copy()
        
        # 转为 torch.Tensor
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        
        return data, label