import os
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Union, Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def list_sgy_files(noisy_dir: str, clean_dir: str) -> Tuple[List[str], List[str]]:
    """
    自动获取两个文件夹下的sgy文件路径，并按文件名排序保证一一对应。
    
    Args:
        noisy_dir (str): 含噪数据文件夹路径
        clean_dir (str): 干净数据文件夹路径

    Returns:
        Tuple[List[str], List[str]]: noisy_files, clean_files
    """
    noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.sgy')])
    clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.sgy')])

    assert len(noisy_files) == len(clean_files), "含噪数据和干净数据文件数不一致！"
    return noisy_files, clean_files

def read_segy(data_dir: str, shotnum: int = 0) -> np.ndarray:
    """
    读取sgy文件中的所有地震数据，并按炮点组织成三维数组。
    
    Args:
        data_dir (str): sgy文件的完整路径。
        shotnum (int): 如果已知炮点数量，可以手动指定，否则函数会自动从文件头中推断。
        
    Returns:
        np.ndarray: 形状为 (shot_num, time, trace_per_shot) 的三维地震数据。
    """
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX) # 所有道的数量
        
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX)) # 炮点数量
        
        len_shot = trace_num // shot_num  # 每个炮点的道数
        time = f.trace[0].shape[0] # 时间采样点数
        
        print(f'开始读取sgy数据: {data_dir}')
        
        data = np.zeros((shot_num, time, len_shot), dtype=np.float32)
        
        for j in range(0, shot_num):
            shot_data = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
            data[j, :, :] = shot_data
        
        # 归一化数据到 [-1, 1]
        data_max = np.abs(data).max()
        if data_max > 0:
            data = data / data_max
            
        return data


class SeismicDenoiseDataset(Dataset):
    """
    地震数据去噪数据集类
    
    将地震数据从 (5, 512, 128) 转换为多个 (128, 128) 的样本
    每炮数据在时间轴上抽稀成4个子样本
    支持数据翻转增强
    """
    
    def __init__(self, 
                 noisy_files: List[str], 
                 clean_files: List[str], 
                 shot_num: int = 5,
                 original_time_samples: int = 512,
                 target_time_samples: int = 128,
                 trace_samples: int = 128,
                 subsample_num: int = 4,
                 use_flip_augmentation: bool = True):
        """
        Args:
            noisy_files (List[str]): 含噪数据文件路径列表
            clean_files (List[str]): 干净数据文件路径列表
            shot_num (int): 每个文件的炮数
            original_time_samples (int): 原始时间采样点数
            target_time_samples (int): 目标时间采样点数
            trace_samples (int): 道数
            subsample_num (int): 每炮数据抽稀成的子样本数量
            use_flip_augmentation (bool): 是否使用翻转数据增强
        """
        assert len(noisy_files) == len(clean_files), "含噪文件和干净文件数量必须相等"
        
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.shot_num = shot_num
        self.original_time_samples = original_time_samples
        self.target_time_samples = target_time_samples
        self.trace_samples = trace_samples
        self.subsample_num = subsample_num
        self.use_flip_augmentation = use_flip_augmentation
        
        # 计算抽稀的步长
        self.stride = (original_time_samples - target_time_samples) // (subsample_num - 1)
        if self.stride <= 0:
            self.stride = 1
            
        # 基础样本数 = 文件数 × 每文件炮数 × 每炮子样本数
        self.base_samples = len(noisy_files) * shot_num * subsample_num
        
        # 如果使用翻转增强，样本数翻倍
        self.total_samples = self.base_samples * (2 if use_flip_augmentation else 1)
        
        print(f"数据集初始化完成:")
        print(f"- 文件对数: {len(noisy_files)}")
        print(f"- 每文件炮数: {shot_num}")
        print(f"- 每炮子样本数: {subsample_num}")
        print(f"- 翻转数据增强: {'开启' if use_flip_augmentation else '关闭'}")
        print(f"- 基础样本数: {self.base_samples}")
        print(f"- 总样本数: {self.total_samples}")
        print(f"- 时间轴抽稀步长: {self.stride}")
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        根据索引返回一个训练样本
        
        Returns:
            tuple: (noisy_sample, clean_sample)
                   每个样本的形状为 (128, 128)
        """
        # 判断是否为翻转样本
        is_flipped = False
        if self.use_flip_augmentation and idx >= self.base_samples:
            is_flipped = True
            idx = idx - self.base_samples  # 获取对应的原始样本索引
        
        # 计算文件索引、炮索引和子样本索引
        file_idx = idx // (self.shot_num * self.subsample_num)
        remaining = idx % (self.shot_num * self.subsample_num)
        shot_idx = remaining // self.subsample_num
        subsample_idx = remaining % self.subsample_num
        
        # 读取对应的含噪和干净数据文件
        noisy_data = read_segy(self.noisy_files[file_idx], shotnum=self.shot_num)
        clean_data = read_segy(self.clean_files[file_idx], shotnum=self.shot_num)
        
        # 提取对应炮的数据 (512, 128)
        noisy_shot = noisy_data[shot_idx]  # (512, 128)
        clean_shot = clean_data[shot_idx]  # (512, 128)
        
        # 在时间轴上进行抽稀，得到 (128, 128)
        start_time = subsample_idx * self.stride
        end_time = start_time + self.target_time_samples
        
        # 确保不超出边界
        if end_time > self.original_time_samples:
            end_time = self.original_time_samples
            start_time = end_time - self.target_time_samples
            
        noisy_sample = noisy_shot[start_time:end_time, :]  # (128, 128)
        clean_sample = clean_shot[start_time:end_time, :]  # (128, 128)
        
        # 数据增强：左右翻转（沿着道的方向翻转）
        if is_flipped:
            noisy_sample = np.fliplr(noisy_sample)  # 左右翻转
            clean_sample = np.fliplr(clean_sample)  # 左右翻转
        
        # 转换为torch张量
        noisy_sample = torch.from_numpy(noisy_sample.astype(np.float32))
        clean_sample = torch.from_numpy(clean_sample.astype(np.float32))
        
        return noisy_sample, clean_sample
    
    def get_sample_info(self, idx):
        """
        获取样本的详细信息，用于调试
        """
        # 判断是否为翻转样本
        is_flipped = False
        original_idx = idx
        if self.use_flip_augmentation and idx >= self.base_samples:
            is_flipped = True
            idx = idx - self.base_samples
        
        file_idx = idx // (self.shot_num * self.subsample_num)
        remaining = idx % (self.shot_num * self.subsample_num)
        shot_idx = remaining // self.subsample_num
        subsample_idx = remaining % self.subsample_num
        
        start_time = subsample_idx * self.stride
        end_time = start_time + self.target_time_samples
        
        if end_time > self.original_time_samples:
            end_time = self.original_time_samples
            start_time = end_time - self.target_time_samples
        
        info = {
            'sample_idx': original_idx,
            'file_idx': file_idx,
            'shot_idx': shot_idx,
            'subsample_idx': subsample_idx,
            'time_range': (start_time, end_time),
            'is_flipped': is_flipped,
            'noisy_file': self.noisy_files[file_idx],
            'clean_file': self.clean_files[file_idx]
        }
        
        return info


def create_dataloader(noisy_files: List[str], 
                     clean_files: List[str], 
                     batch_size: int = 16,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     **dataset_kwargs):
    """
    创建数据加载器
    """
    dataset = SeismicDenoiseDataset(noisy_files, clean_files, **dataset_kwargs)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# 使用示例
if __name__ == "__main__":
    # 示例文件路径列表
    noisy_dir = "D:\\桌面\\面波压制\\合成记录\\data\\model\\noisy"
    clean_dir = "D:\\桌面\\面波压制\\合成记录\\data\\model\\clean"

    # 自动获取文件路径
    noisy_files, clean_files = list_sgy_files(noisy_dir, clean_dir)
    
    # 创建数据集
    dataset = SeismicDenoiseDataset(
        noisy_files=noisy_files,
        clean_files=clean_files,
        shot_num=5,
        original_time_samples=512,
        target_time_samples=128,
        trace_samples=128,
        subsample_num=4,
        use_flip_augmentation=True  # 启用翻转数据增强
    )
    
    # 测试数据集
    print(f"\n数据集大小: {len(dataset)}")
    print(f"基础样本数: {dataset.base_samples}")
    if dataset.use_flip_augmentation:
        print(f"翻转增强样本数: {dataset.base_samples}")
    
    # 获取第一个样本（原始样本）
    noisy_sample, clean_sample = dataset[0]
    print(f"\n原始样本:")
    print(f"- 含噪样本形状: {noisy_sample.shape}")
    print(f"- 干净样本形状: {clean_sample.shape}")
    
    # 获取翻转样本（如果启用了数据增强）
    if dataset.use_flip_augmentation:
        flip_idx = dataset.base_samples  # 第一个翻转样本的索引
        noisy_flip, clean_flip = dataset[flip_idx]
        print(f"\n翻转样本:")
        print(f"- 含噪样本形状: {noisy_flip.shape}")
        print(f"- 干净样本形状: {clean_flip.shape}")
    
    # 获取样本信息
    print(f"\n原始样本信息:")
    info = dataset.get_sample_info(0)
    for key, value in info.items():
        print(f"- {key}: {value}")
    
    if dataset.use_flip_augmentation:
        print(f"\n翻转样本信息:")
        flip_info = dataset.get_sample_info(dataset.base_samples)
        for key, value in flip_info.items():
            print(f"- {key}: {value}")
    
    # 展示数据集统计信息
    print(f"\n=== 数据集统计信息 ===")
    print(f"文件对数量: {len(dataset.noisy_files)}")
    print(f"每文件炮数: {dataset.shot_num}")
    print(f"每炮子样本数: {dataset.subsample_num}")
    print(f"原始时间采样点: {dataset.original_time_samples}")
    print(f"目标时间采样点: {dataset.target_time_samples}")
    print(f"道数: {dataset.trace_samples}")
    print(f"数据增强: {'启用' if dataset.use_flip_augmentation else '禁用'}")
    print(f"基础样本总数: {dataset.base_samples}")
    print(f"最终样本总数: {dataset.total_samples}")
    
    # 计算样本分布
    samples_per_file = dataset.shot_num * dataset.subsample_num
    if dataset.use_flip_augmentation:
        samples_per_file *= 2
    print(f"每文件样本数: {samples_per_file}")
    print(f"预期总样本数: {len(dataset.noisy_files) * samples_per_file}")
    
    # 验证数据集大小计算
    assert len(dataset) == len(dataset.noisy_files) * samples_per_file, "数据集大小计算错误!"
    print(f"✓ 数据集大小验证通过")
    
    # 创建数据加载器
    dataloader = create_dataloader(
        noisy_files=noisy_files,
        clean_files=clean_files,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    
    # 测试数据加载器
    for batch_idx, (noisy_batch, clean_batch) in enumerate(dataloader):
        print(f"批次 {batch_idx}: 含噪批次形状 {noisy_batch.shape}, 干净批次形状 {clean_batch.shape}")
        if batch_idx >= 2:  # 只测试前几个批次
            break
    
    # 可视化示例
    def visualize_sample(dataset, sample_idx=0, show_flip_comparison=True):
        """可视化样本，支持显示翻转对比"""
        noisy_sample, clean_sample = dataset[sample_idx]
        info = dataset.get_sample_info(sample_idx)
        
        # 如果启用了翻转增强且要求显示对比
        if show_flip_comparison and dataset.use_flip_augmentation:
            # 获取对应的翻转样本
            if sample_idx < dataset.base_samples:
                flip_idx = sample_idx + dataset.base_samples
            else:
                flip_idx = sample_idx - dataset.base_samples
            
            noisy_flip, clean_flip = dataset[flip_idx]
            flip_info = dataset.get_sample_info(flip_idx)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 原始含噪数据
            im1 = axes[0, 0].imshow(noisy_sample.numpy(), aspect='auto', cmap='seismic')
            axes[0, 0].set_title(f'原始含噪数据 (样本 {sample_idx})')
            axes[0, 0].set_xlabel('道数')
            axes[0, 0].set_ylabel('时间采样点')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # 原始干净数据
            im2 = axes[0, 1].imshow(clean_sample.numpy(), aspect='auto', cmap='seismic')
            axes[0, 1].set_title(f'原始干净数据 (样本 {sample_idx})')
            axes[0, 1].set_xlabel('道数')
            axes[0, 1].set_ylabel('时间采样点')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # 翻转含噪数据
            im3 = axes[1, 0].imshow(noisy_flip.numpy(), aspect='auto', cmap='seismic')
            axes[1, 0].set_title(f'翻转含噪数据 (样本 {flip_idx})')
            axes[1, 0].set_xlabel('道数')
            axes[1, 0].set_ylabel('时间采样点')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # 翻转干净数据
            im4 = axes[1, 1].imshow(clean_flip.numpy(), aspect='auto', cmap='seismic')
            axes[1, 1].set_title(f'翻转干净数据 (样本 {flip_idx})')
            axes[1, 1].set_xlabel('道数')
            axes[1, 1].set_ylabel('时间采样点')
            plt.colorbar(im4, ax=axes[1, 1])
            
            print(f"原始样本信息: {info}")
            print(f"翻转样本信息: {flip_info}")
            
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 含噪数据
            im1 = axes[0].imshow(noisy_sample.numpy(), aspect='auto', cmap='seismic')
            axes[0].set_title(f'含噪数据 (样本 {sample_idx})')
            axes[0].set_xlabel('道数')
            axes[0].set_ylabel('时间采样点')
            plt.colorbar(im1, ax=axes[0])
            
            # 干净数据
            im2 = axes[1].imshow(clean_sample.numpy(), aspect='auto', cmap='seismic')
            axes[1].set_title(f'干净数据 (样本 {sample_idx})')
            axes[1].set_xlabel('道数')
            axes[1].set_ylabel('时间采样点')
            plt.colorbar(im2, ax=axes[1])
            
            print(f"样本详细信息: {info}")
        
        plt.tight_layout()
        plt.show()
    
    # 可视化第一个样本和翻转对比
    # visualize_sample(dataset, 0, show_flip_comparison=True)