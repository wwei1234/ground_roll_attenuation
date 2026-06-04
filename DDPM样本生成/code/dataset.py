# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class SeismicPix2PixDataset(Dataset):
    """
    配对数据集：每一个炮集文件对应一个同名（或按规则命名）的 mask 文件。

    Domain A（输入）：面波位置 mask，二值图，作为生成器的条件
    Domain B（目标）：含面波的真实炮集，生成器的学习目标

    目录结构要求：
        gather_dir/
            real_data_gather001.npy
            real_data_gather002.npy
            ...
        mask_dir/
            mask_001.npy   （由 map_mask_filename 函数自动匹配）
            mask_002.npy
            ...
    """

    def __init__(self, gather_dir, mask_dir, target_shape=(512, 256), augment=True):
        """
        gather_dir   : 含面波炮集文件夹路径
        mask_dir     : mask 文件夹路径
        target_shape : (time_samples, num_traces)，统一 resize 到此尺寸，需为 32 的倍数
        augment      : 是否开启数据增强
        """
        self.gather_dir   = gather_dir
        self.mask_dir     = mask_dir
        self.target_shape = target_shape
        self.augment      = augment

        # 枚举所有炮集文件并寻找配对的 mask
        self.pairs = []
        gather_files = sorted([f for f in os.listdir(gather_dir) if f.endswith('.npy')])

        for gf in gather_files:
            mask_name = self._map_mask_filename(gf)
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.pairs.append((os.path.join(gather_dir, gf), mask_path))
            else:
                print(f"[警告] 未找到配对 mask，跳过: {gf} → {mask_name}")

        assert len(self.pairs) > 0, \
            f"未找到任何配对样本，请检查 gather_dir 和 mask_dir 的文件命名规则。"
        print(f"[数据集] 共找到 {len(self.pairs)} 对配对样本")

    @staticmethod
    def _map_mask_filename(gather_filename):
        """
        根据炮集文件名推断对应的 mask 文件名。
        规则：提取文件名中最后一段数字，生成 mask_XXX.npy。
        示例：real_data_gather185.npy → mask_185.npy
        如需其他命名规则，在此处修改。
        """
        import re
        base   = os.path.splitext(gather_filename)[0]
        digits = re.findall(r'\d+', base)
        if digits:
            num = int(digits[-1])
            return f"mask_{num:03d}.npy"
        return gather_filename  # 兜底：同名

    @staticmethod
    def _resize(data, target_shape, order=1):
        """双线性插值 resize，避免引入额外依赖"""
        h, w   = data.shape
        th, tw = target_shape
        if (h, w) == (th, tw):
            return data
        return zoom(data, (th / h, tw / w), order=1).astype(np.float32)

    def _normalize_gather(self, data):
        """炮集：3σ 截断后归一化到 [-1, 1]"""
        sigma = np.std(data)
        if sigma > 1e-8:
            data = np.clip(data, -3 * sigma, 3 * sigma) / (3 * sigma)
        else:
            data = np.zeros_like(data)
        return data

    def _normalize_mask(self, data):
        """mask：二值 0/1 → [-1, 1]"""
        data = (data > 0.5).astype(np.float32)
        return data * 2.0 - 1.0

    def _augment(self, gather, mask):
        """
        数据增强（炮集与 mask 同步操作，保证配对不变）：
          1. 随机振幅缩放：仅作用于炮集，mask 不变
          2. 随机加高斯噪声：仅作用于炮集，mask 不变
          3. 随机道序翻转（左右翻转）：炮集与 mask 同步
        """
        # 1. 随机振幅缩放 [0.8, 1.2]
        scale   = np.random.uniform(0.8, 1.2)
        gather  = np.clip(gather * scale, -1.0, 1.0)

        # 2. 随机叠加高斯噪声（SNR 约 20dB）
        noise_std = np.random.uniform(0.0, 0.05)
        gather    = np.clip(gather + np.random.randn(*gather.shape).astype(np.float32) * noise_std,
                            -1.0, 1.0)

        # 3. 随机道序翻转（50% 概率）
        if np.random.rand() > 0.5:
            gather = gather[:, ::-1].copy()
            mask   = mask[:, ::-1].copy()

        return gather, mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gather_path, mask_path = self.pairs[idx]

        # 加载
        gather = np.load(gather_path).astype(np.float32)
        mask   = np.load(mask_path).astype(np.float32)

        # 取第一通道（兼容 3D 数组）
        if gather.ndim == 3:
            gather = gather[0]
        if mask.ndim == 3:
            mask = mask[0]

        # Resize
        gather = self._resize(gather, self.target_shape, order=1)  # 炮集用双线性插值
        mask   = self._resize(mask,   self.target_shape, order=0)  # mask 用最近邻插值保持二值特性

        # 归一化
        gather = self._normalize_gather(gather)
        mask   = self._normalize_mask(mask)

        # 数据增强
        if self.augment:
            gather, mask = self._augment(gather, mask)

        # 转 Tensor，增加通道维度 → (1, H, W)
        gather_t = torch.from_numpy(gather).unsqueeze(0)
        mask_t   = torch.from_numpy(mask).unsqueeze(0)

        return {
            'A': mask_t,          # 输入条件：mask，(1, H, W)，值域 [-1, 1]
            'B': gather_t,        # 生成目标：炮集，(1, H, W)，值域 [-1, 1]
            'A_path': mask_path,
            'B_path': gather_path,
        }