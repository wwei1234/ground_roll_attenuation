"""
基于手动标注面波掩码的地震数据集
==================================
核心逻辑：
  1. 面波区域（sw_mask）：手动标注，输入替换为高斯噪声，不参与损失
  2. 连续扇形区域（fan_mask）：手动标注，输入替换为高斯噪声，label 为原始数据，参与损失
  3. 数据增强：左右翻转，sw_mask 和 fan_mask 同步翻转

文件结构：
  gather_dir/   real_data_gather000.npy ~ real_data_gather191.npy
  mask_dir/     mask_000.npy            ~ mask_191.npy
  fan_mask_dir/ fan_mask_000.npy        ~ fan_mask_191.npy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from matplotlib import rcParams


# ==============================
# 工具函数
# ==============================
def normalize_traces_per_trace(shot_gather: np.ndarray) -> np.ndarray:
    """逐道归一化（每道除以该道最大绝对值）"""
    normalized = shot_gather.copy()
    for i in range(normalized.shape[1]):
        trace = normalized[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized[:, i] = trace / max_amp
    return normalized


# ==============================
# Dataset 类
# ==============================
class ManualMaskSeismicDataset(Dataset):
    """
    Parameters
    ----------
    gather_dir : str
        炮集 npy 文件所在目录
    mask_dir : str
        面波区域手动掩码目录
    fan_mask_dir : str
        连续扇形手动掩码目录
        若某炮 fan_mask 不存在则跳过该炮
    total : int
        炮集总数，默认 192
    prefix_gather : str
        炮集文件名前缀，默认 "real_data_gather"
    prefix_mask : str
        面波掩码文件名前缀，默认 "mask_"
    prefix_fan_mask : str
        扇形掩码文件名前缀，默认 "fan_mask_"
    max_time_samples : int
        最多使用的时间采样点数，默认 None（不截断）
    mask_k : float
        高斯噪声幅度上限，默认 0.5
    enable_augmentation : bool
        是否启用左右翻转增强，默认 True
    loss_on_full_non_sw : bool
        True  → 损失区域为全部非面波区域（推荐）
        False → 损失区域仅为 fan_mask 位置
        默认 True
    normalize : bool
        是否逐道归一化，默认 True
    """

    def __init__(
        self,
        gather_dir: str,
        mask_dir: str,
        fan_mask_dir: str,
        total: int = 192,
        prefix_gather: str = "real_data_gather",
        prefix_mask: str = "mask_",
        prefix_fan_mask: str = "fan_mask_",
        max_time_samples: int = None,
        mask_k: float = 0.5,
        enable_augmentation: bool = True,
        loss_on_full_non_sw: bool = True,
        normalize: bool = True,
    ):
        self.gather_dir          = gather_dir
        self.mask_dir            = mask_dir
        self.fan_mask_dir        = fan_mask_dir
        self.total               = total
        self.prefix_gather       = prefix_gather
        self.prefix_mask         = prefix_mask
        self.prefix_fan_mask     = prefix_fan_mask
        self.max_time_samples    = max_time_samples
        self.mask_k              = mask_k
        self.enable_augmentation = enable_augmentation
        self.loss_on_full_non_sw = loss_on_full_non_sw
        self.normalize           = normalize

        print("=" * 65)
        print("正在加载手动标注面波掩码数据集...")
        print("=" * 65)
        print(f"炮集目录      : {gather_dir}")
        print(f"面波掩码目录  : {mask_dir}")
        print(f"扇形掩码目录  : {fan_mask_dir}")
        print(f"总炮数        : {total}")

        self.gathers       = []
        self.sw_masks      = []
        self.fan_masks     = []
        self.valid_indices = []

        skipped_no_fan = 0

        for idx in range(total):
            num_str = f"{idx:03d}"
            g_path  = os.path.join(gather_dir,   f"{prefix_gather}{num_str}.npy")
            m_path  = os.path.join(mask_dir,     f"{prefix_mask}{num_str}.npy")
            fm_path = os.path.join(fan_mask_dir, f"{prefix_fan_mask}{num_str}.npy")

            if not os.path.exists(g_path):
                print(f"  [跳过] 炮集不存在: {g_path}")
                continue
            if not os.path.exists(m_path):
                print(f"  [跳过] 面波掩码不存在: {m_path}")
                continue
            if not os.path.exists(fm_path):
                # fan_mask 不存在则跳过该炮（不再退化为离散盲扇）
                skipped_no_fan += 1
                continue

            gather = np.load(g_path).astype(np.float32)

            mask = np.load(m_path, allow_pickle=True)
            if mask.ndim == 0:
                mask = mask.item()
            if not isinstance(mask, np.ndarray):
                print(f"  [跳过] 面波掩码内容异常: {m_path}")
                continue
            mask = mask.astype(np.uint8)

            fm = np.load(fm_path, allow_pickle=True)
            if fm.ndim == 0:
                fm = fm.item()
            if not isinstance(fm, np.ndarray):
                print(f"  [跳过] 扇形掩码内容异常: {fm_path}")
                continue
            fm = fm.astype(np.uint8)

            if gather.ndim != 2 or mask.ndim != 2 or fm.ndim != 2:
                print(f"  [跳过] 编号 {num_str} 数据维度不是 2D")
                continue
            if gather.shape != mask.shape or gather.shape != fm.shape:
                print(f"  [跳过] 编号 {num_str} 形状不一致: "
                      f"炮集{gather.shape} 面波掩码{mask.shape} 扇形掩码{fm.shape}")
                continue

            if max_time_samples is not None:
                gather = gather[:max_time_samples, :]
                mask   = mask[:max_time_samples, :]
                fm     = fm[:max_time_samples, :]

            if normalize:
                gather = normalize_traces_per_trace(gather)

            self.gathers.append(gather)
            self.sw_masks.append(mask)
            self.fan_masks.append(fm)
            self.valid_indices.append(idx)

        self.original_count = len(self.gathers)
        if self.original_count == 0:
            raise RuntimeError("未成功加载任何炮集，请检查目录和文件名配置。")

        self.gathers   = np.array(self.gathers,   dtype=np.float32)
        self.sw_masks  = np.array(self.sw_masks,  dtype=np.uint8)
        self.fan_masks = np.array(self.fan_masks, dtype=np.uint8)

        self.n_time, self.n_trace = self.gathers.shape[1], self.gathers.shape[2]

        # ---------- 左右翻转数据增强 ----------
        if enable_augmentation:
            self.gathers   = np.concatenate(
                [self.gathers,   np.flip(self.gathers,   axis=2).copy()], axis=0)
            self.sw_masks  = np.concatenate(
                [self.sw_masks,  np.flip(self.sw_masks,  axis=2).copy()], axis=0)
            self.fan_masks = np.concatenate(
                [self.fan_masks, np.flip(self.fan_masks, axis=2).copy()], axis=0)

        self.total_count = len(self.gathers)

        print(f"\n成功加载     : {self.original_count} 炮")
        if skipped_no_fan:
            print(f"  跳过（无扇形掩码）: {skipped_no_fan} 炮")
        print(f"数据形状     : 时间 {self.n_time} × 道数 {self.n_trace}")
        print(f"数据增强     : {'启用（翻转）' if enable_augmentation else '关闭'}")
        print(f"总样本数     : {self.total_count}")
        print(f"\n掩码策略:")
        print(f"  高斯幅度范围 : [-{mask_k}, {mask_k}]")
        print(f"  损失区域     : {'全部非面波区域' if loss_on_full_non_sw else '仅扇形掩码位置'}")
        print("=" * 65)

    # ==============================
    # 辅助方法
    # ==============================
    def is_flipped(self, idx: int) -> bool:
        return self.enable_augmentation and idx >= self.original_count

    def original_idx(self, idx: int) -> int:
        if self.is_flipped(idx):
            return self.valid_indices[idx - self.original_count]
        return self.valid_indices[idx]

    def _generate_sample(self, idx: int):
        gather   = self.gathers[idx].copy()
        sw_mask  = self.sw_masks[idx].astype(bool)
        fan_mask = self.fan_masks[idx].astype(bool)

        sw_valid     = sw_mask
        non_sw_valid = ~sw_mask

        # fan_mask 限制在非面波区域内（防止标注越界）
        fan_valid = fan_mask & non_sw_valid

        # 高斯噪声
        gaussian = np.random.normal(
            0, self.mask_k / 3.0,
            size=(self.n_time, self.n_trace)
        ).astype(np.float32)
        gaussian = np.clip(gaussian, -self.mask_k, self.mask_k)

        # ---------- 构建 Input ----------
        # 面波区域     → 高斯噪声
        # 扇形掩码区域 → 高斯噪声
        # 其余非面波   → 原始数据
        input_data = gather.copy()
        input_data[sw_valid]  = gaussian[sw_valid]
        input_data[fan_valid] = gaussian[fan_valid]

        # ---------- 构建 Label ----------
        # 面波区域     → 高斯噪声（不参与损失）
        # 其余所有区域 → 原始数据（含扇形位置，有 ground truth）✅
        label_data = gather.copy()
        label_data[sw_valid] = gaussian[sw_valid]

        # ---------- 损失区域 ----------
        if self.loss_on_full_non_sw:
            loss_region = non_sw_valid
        else:
            loss_region = fan_valid

        return {
            'input_data':        input_data,
            'label_data':        label_data,
            'original_data':     gather,
            'surface_wave_mask': sw_mask.astype(np.uint8),
            'fan_mask':          fan_valid.astype(np.uint8),
            'non_sw_region':     non_sw_valid.astype(np.uint8),
            'loss_region':       loss_region.astype(np.uint8),
            'shot_idx':          idx,
            'original_shot_idx': self.original_idx(idx),
            'is_flipped':        self.is_flipped(idx),
        }

    # ==============================
    # Dataset 接口
    # ==============================
    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        return self._generate_sample(idx)

    # ==============================
    # 可视化
    # ==============================
    def visualize_sample(self, idx: int, figsize=(21, 10)):
        s = self._generate_sample(idx)

        input_data    = s['input_data']
        label_data    = s['label_data']
        original_data = s['original_data']
        sw_mask       = s['surface_wave_mask']
        fan_mask      = s['fan_mask']
        loss_region   = s['loss_region']
        is_flip       = s['is_flipped']
        orig_idx      = s['original_shot_idx']

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        vmax = np.percentile(np.abs(original_data), 99) or 1.0
        kw_seis = dict(aspect='auto', cmap='seismic',
                       vmin=-vmax, vmax=vmax, interpolation='bilinear')
        kw_mask = dict(aspect='auto', interpolation='nearest', vmin=0, vmax=1)

        shot_center = getattr(self, 'shot_center', self.n_trace // 2)
        if is_flip:
            shot_center = self.n_trace - 1 - shot_center

        def add_shot_line(ax):
            ax.axvline(shot_center, color='lime', linestyle='--', linewidth=1.5)

        im = axes[0].imshow(original_data, **kw_seis)
        axes[0].set_title(f'1. 原始数据  |  {"翻转" if is_flip else "原始"}样本 #{orig_idx}',
                          fontweight='bold')
        add_shot_line(axes[0]); plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        im = axes[1].imshow(sw_mask, cmap='Reds', **kw_mask)
        axes[1].set_title('2. 面波区域（手动标注）', color='red')
        add_shot_line(axes[1]); plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        im = axes[2].imshow(fan_mask, cmap='Blues', **kw_mask)
        axes[2].set_title('3. 扇形掩码（手动标注）', color='blue')
        add_shot_line(axes[2]); plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        im = axes[3].imshow(input_data, **kw_seis)
        axes[3].set_title('4. 网络输入 (Input)', fontweight='bold', color='blue')
        add_shot_line(axes[3]); plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        im = axes[4].imshow(label_data, **kw_seis)
        axes[4].set_title('5. 网络标签 (Label)', fontweight='bold', color='green')
        add_shot_line(axes[4]); plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

        im = axes[5].imshow(loss_region, cmap='Oranges', **kw_mask)
        loss_title = ('6. 损失区域（全部非面波）' if self.loss_on_full_non_sw
                      else '6. 损失区域（仅扇形掩码）')
        axes[5].set_title(loss_title, fontweight='bold', color='darkorange')
        add_shot_line(axes[5]); plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel('道号'); ax.set_ylabel('时间采样点')

        total_px   = self.n_time * self.n_trace
        sw_ratio   = sw_mask.sum()  / total_px * 100
        fan_ratio  = fan_mask.sum() / total_px * 100
        fig.suptitle(
            f'样本 {idx}  |  面波占比: {sw_ratio:.2f}%  |  '
            f'扇形掩码占比: {fan_ratio:.2f}%  |  '
            f'高斯范围: [-{self.mask_k}, {self.mask_k}]',
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        return fig


# ==============================
# ★ 使用示例
# ==============================
if __name__ == "__main__":
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

    dataset = ManualMaskSeismicDataset(
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
    dataset.shot_center = 85

    print(f"数据集大小: {len(dataset)}")
    fig = dataset.visualize_sample(139)
    plt.show()