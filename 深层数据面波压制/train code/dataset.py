"""
基于手动标注面波掩码的地震数据集
==================================
与原 SeismicMaskDataset 的主要区别：
  1. 面波区域来自手动标注的 0/1 掩码文件，不再自动识别
  2. 去掉了直达波去除步骤
  3. 非面波区域的高斯掩码生成逻辑与原版完全一致
  4. 支持左右翻转数据增强
  5. 盲扇掩码模式可选：
       'half_line' → 仅反向延长线（从选中点继续延伸，远离炮点方向）
       'full_line' → 整条直线（炮点到选中点再到图像边界的完整线）

文件结构假设：
  gather_dir/  real_data_gather000.npy ~ real_data_gather191.npy   (炮集，shape: time × trace)
  mask_dir/    mask_000.npy            ~ mask_191.npy              (手动掩码，shape: time × trace，值 0/1)
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
    基于手动标注面波掩码的地震炮集数据集。

    Parameters
    ----------
    gather_dir : str
        炮集 npy 文件所在目录
    mask_dir : str
        手动标注掩码 npy 文件所在目录
    total : int
        炮集总数，编号范围 000 ~ (total-1)，默认 192
    prefix_gather : str
        炮集文件名前缀，默认 "real_data_gather"
    prefix_mask : str
        掩码文件名前缀，默认 "mask_"
    max_time_samples : int
        最多使用的时间采样点数，超出部分截断，默认 None（不截断）
    non_sw_density_range : tuple
        非面波区域高斯掩码密度百分比范围 (min%, max%)，默认 (5, 15)
    mask_k : float
        高斯掩码的幅度上限，掩码值裁剪到 [-mask_k, mask_k]，默认 0.5
    enable_augmentation : bool
        是否启用左右翻转数据增强，默认 True
    loss_on_full_non_sw : bool
        True → 损失计算区域为全部非面波区域；
        False → 损失计算区域仅为非面波掩码位置，默认 True
    normalize : bool
        是否对每个炮集做逐道归一化，默认 True
    blind_spot_mode : str
        盲扇掩码模式：
          'half_line' → 仅反向延长线（从选中点沿炮点→选中点方向继续延伸）
          'full_line' → 整条直线（从炮点出发经过选中点直到图像边界）
        默认 'half_line'
    """

    def __init__(
        self,
        gather_dir: str,
        mask_dir: str,
        total: int = 192,
        prefix_gather: str = "real_data_gather",
        prefix_mask: str = "mask_",
        max_time_samples: int = None,
        non_sw_density_range: tuple = (5, 15),
        mask_k: float = 0.5,
        enable_augmentation: bool = True,
        loss_on_full_non_sw: bool = True,
        normalize: bool = True,
        blind_spot_mode: str = 'half_line',
    ):
        assert blind_spot_mode in ('half_line', 'full_line'), \
            "blind_spot_mode 必须为 'half_line' 或 'full_line'"

        self.gather_dir = gather_dir
        self.mask_dir = mask_dir
        self.total = total
        self.prefix_gather = prefix_gather
        self.prefix_mask = prefix_mask
        self.max_time_samples = max_time_samples
        self.non_sw_density_range = non_sw_density_range
        self.mask_k = mask_k
        self.enable_augmentation = enable_augmentation
        self.loss_on_full_non_sw = loss_on_full_non_sw
        self.normalize = normalize
        self.blind_spot_mode = blind_spot_mode

        # ---------- 加载数据 ----------
        print("=" * 65)
        print("正在加载手动标注面波掩码数据集...")
        print("=" * 65)
        print(f"炮集目录  : {gather_dir}")
        print(f"掩码目录  : {mask_dir}")
        print(f"总炮数    : {total}")
        print(f"盲扇模式  : {'仅反向延长线' if blind_spot_mode == 'half_line' else '整条直线'}")

        self.gathers = []
        self.sw_masks = []
        self.valid_indices = []

        for idx in range(total):
            num_str = f"{idx:03d}"
            g_path = os.path.join(gather_dir, f"{prefix_gather}{num_str}.npy")
            m_path = os.path.join(mask_dir,   f"{prefix_mask}{num_str}.npy")

            if not os.path.exists(g_path):
                print(f"  [跳过] 炮集文件不存在: {g_path}")
                continue
            if not os.path.exists(m_path):
                print(f"  [跳过] 掩码文件不存在: {m_path}")
                continue

            gather = np.load(g_path).astype(np.float32)
            mask   = np.load(m_path, allow_pickle=True)

            # 处理 allow_pickle 可能返回对象数组的情况
            if mask.ndim == 0:
                mask = mask.item()
            if not isinstance(mask, np.ndarray):
                print(f"  [跳过] 掩码文件内容异常: {m_path}")
                continue
            mask = mask.astype(np.uint8)

            # 维度检查
            if gather.ndim != 2 or mask.ndim != 2:
                print(f"  [跳过] 编号 {num_str} 数据维度不是 2D")
                continue
            if gather.shape != mask.shape:
                print(f"  [跳过] 编号 {num_str} 炮集 {gather.shape} 与掩码 {mask.shape} 形状不一致")
                continue

            # 截断时间采样点
            if max_time_samples is not None:
                gather = gather[:max_time_samples, :]
                mask   = mask[:max_time_samples, :]

            # 逐道归一化
            if normalize:
                gather = normalize_traces_per_trace(gather)

            self.gathers.append(gather)
            self.sw_masks.append(mask)
            self.valid_indices.append(idx)

        self.original_count = len(self.gathers)
        if self.original_count == 0:
            raise RuntimeError("未成功加载任何炮集，请检查目录和文件名配置。")

        # 数组化
        self.gathers  = np.array(self.gathers,  dtype=np.float32)
        self.sw_masks = np.array(self.sw_masks, dtype=np.uint8)

        self.n_time, self.n_trace = self.gathers.shape[1], self.gathers.shape[2]

        # 数据增强：在列表末尾追加翻转样本
        if enable_augmentation:
            flipped_gathers  = np.flip(self.gathers,  axis=2).copy()
            flipped_sw_masks = np.flip(self.sw_masks, axis=2).copy()
            self.gathers  = np.concatenate([self.gathers,  flipped_gathers],  axis=0)
            self.sw_masks = np.concatenate([self.sw_masks, flipped_sw_masks], axis=0)

        self.total_count = len(self.gathers)

        # ---------- 打印摘要 ----------
        print(f"\n成功加载   : {self.original_count} 炮")
        print(f"数据形状   : 时间 {self.n_time} × 道数 {self.n_trace}")
        print(f"数据增强   : {'启用（翻转）' if enable_augmentation else '关闭'}")
        print(f"总样本数   : {self.total_count}")
        print(f"\n高斯掩码策略:")
        print(f"  幅度范围  : [-{mask_k}, {mask_k}]")
        print(f"  非面波密度: {non_sw_density_range[0]}% ~ {non_sw_density_range[1]}%")
        print(f"  盲扇模式  : {'仅反向延长线 (half_line)' if blind_spot_mode == 'half_line' else '整条直线 (full_line)'}")
        print(f"  损失区域  : {'全部非面波区域' if loss_on_full_non_sw else '仅非面波掩码位置'}")
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

    def _get_line_pixels(self, trace_idx: int, time_idx: int, shot_center: int):
        """
        根据 blind_spot_mode 返回需要设置盲扇掩码的像素坐标列表。

        half_line：从选中点 (time_idx, trace_idx) 出发，
                   沿炮点→选中点方向继续延伸至图像边界（反向延长线）。

        full_line：从炮点 (0, shot_center) 出发，
                   经过选中点，一直延伸至图像边界（完整直线）。
        """
        t0, x0 = 0, shot_center       # 炮点
        t1, x1 = time_idx, trace_idx  # 选中点

        dt = t1 - t0
        dx = x1 - x0

        if dt == 0 and dx == 0:
            return [(t1, x1)]

        # 归一化步长（使最大分量步长为1，避免跳格）
        length = max(abs(dt), abs(dx))
        step_t = dt / length
        step_x = dx / length

        pixels = []

        if self.blind_spot_mode == 'half_line':
            # 从选中点出发，沿炮点→选中点方向延伸
            t, x = float(t1), float(x1)
            while 0 <= int(round(t)) < self.n_time and 0 <= int(round(x)) < self.n_trace:
                pixels.append((int(round(t)), int(round(x))))
                t += step_t
                x += step_x

        else:  # full_line
            # 从炮点出发，沿炮点→选中点方向延伸至图像边界
            t, x = float(t0), float(x0)
            while 0 <= int(round(t)) < self.n_time and 0 <= int(round(x)) < self.n_trace:
                pixels.append((int(round(t)), int(round(x))))
                t += step_t
                x += step_x

        return pixels

    def _generate_sample(self, idx: int):
        """
        生成一个训练样本，返回所有中间结果。
        """
        gather  = self.gathers[idx].copy()         # (T, X)
        sw_mask = self.sw_masks[idx].astype(bool)  # (T, X)

        valid_region = np.ones((self.n_time, self.n_trace), dtype=bool)
        sw_valid     = sw_mask & valid_region
        non_sw_valid = (~sw_mask) & valid_region

        # ---------- 生成高斯噪声图 ----------
        gaussian = np.random.normal(
            0, self.mask_k / 3.0,
            size=(self.n_time, self.n_trace)
        ).astype(np.float32)
        gaussian = np.clip(gaussian, -self.mask_k, self.mask_k)

        # ---------- 炮点位置 ----------
        if self.is_flipped(idx):
            shot_center = self.n_trace - 1 - getattr(self, 'shot_center', self.n_trace // 2)
        else:
            shot_center = getattr(self, 'shot_center', self.n_trace // 2)

        # ---------- 生成盲扇掩码位置 ----------
        non_sw_mask_pos = np.zeros((self.n_time, self.n_trace), dtype=bool)
        valid_traces = np.where(np.any(non_sw_valid, axis=0))[0]

        for tr in valid_traces:
            times_in_trace = np.where(non_sw_valid[:, tr])[0]
            if len(times_in_trace) == 0:
                continue

            density  = np.random.uniform(*self.non_sw_density_range) / 100.0
            n_active = max(1, int(len(times_in_trace) * density))
            n_active = min(n_active, len(times_in_trace))

            selected = np.random.choice(times_in_trace, size=n_active, replace=False)
            for t_idx in selected:
                pixels = self._get_line_pixels(tr, t_idx, shot_center)
                for t, x in pixels:
                    if non_sw_valid[t, x]:
                        non_sw_mask_pos[t, x] = True

        # ---------- 构建 Input ----------
        # 面波区域      → 高斯噪声
        # 盲扇掩码位置  → 高斯噪声（网络需要预测这里）
        # 其余非面波区域 → 原始数据
        input_data = gather.copy()
        input_data[sw_valid]        = gaussian[sw_valid]
        input_data[non_sw_mask_pos] = gaussian[non_sw_mask_pos]

        # ---------- 构建 Label ----------
        # 面波区域      → 高斯噪声（与 Input 一致，不参与损失）
        # 其余所有区域  → 原始数据
        label_data = gather.copy()
        label_data[sw_valid] = gaussian[sw_valid]

        # ---------- 损失区域 ----------
        if self.loss_on_full_non_sw:
            loss_region = non_sw_valid
        else:
            loss_region = non_sw_mask_pos

        return {
            'input_data':            input_data,
            'label_data':            label_data,
            'original_data':         gather,
            'surface_wave_mask':     sw_mask.astype(np.uint8),
            'non_sw_mask_positions': non_sw_mask_pos.astype(np.uint8),
            'non_sw_region':         non_sw_valid.astype(np.uint8),
            'loss_region':           loss_region.astype(np.uint8),
            'shot_idx':              idx,
            'original_shot_idx':     self.original_idx(idx),
            'is_flipped':            self.is_flipped(idx),
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
    def visualize_sample(self, idx: int, figsize=(18, 10)):
        """绘制 6 张核心子图。"""
        s = self._generate_sample(idx)

        input_data    = s['input_data']
        label_data    = s['label_data']
        original_data = s['original_data']
        sw_mask       = s['surface_wave_mask']
        non_sw_pos    = s['non_sw_mask_positions']
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

        mode_label = '反向延长线' if self.blind_spot_mode == 'half_line' else '整条直线'

        im = axes[0].imshow(original_data, **kw_seis)
        axes[0].set_title(f'1. 原始数据  |  {"翻转" if is_flip else "原始"}样本 #{orig_idx}',
                          fontweight='bold')
        add_shot_line(axes[0]); plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        im = axes[1].imshow(sw_mask, cmap='Reds', **kw_mask)
        axes[1].set_title('2. 面波区域（手动标注）', color='red')
        add_shot_line(axes[1]); plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        im = axes[2].imshow(non_sw_pos, cmap='Blues', **kw_mask)
        axes[2].set_title(f'3. 盲扇掩码位置（{mode_label}）', color='blue')
        add_shot_line(axes[2]); plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        im = axes[3].imshow(input_data, **kw_seis)
        axes[3].set_title('4. 网络输入 (Input)', fontweight='bold', color='blue')
        add_shot_line(axes[3]); plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        im = axes[4].imshow(label_data, **kw_seis)
        axes[4].set_title('5. 网络标签 (Label)', fontweight='bold', color='green')
        add_shot_line(axes[4]); plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

        im = axes[5].imshow(loss_region, cmap='Oranges', **kw_mask)
        loss_title = ('6. 损失区域（全部非面波）' if self.loss_on_full_non_sw
                      else '6. 损失区域（仅掩码位置）')
        axes[5].set_title(loss_title, fontweight='bold', color='darkorange')
        add_shot_line(axes[5]); plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel('道号'); ax.set_ylabel('时间采样点')

        total_px   = self.n_time * self.n_trace
        sw_ratio   = sw_mask.sum()    / total_px * 100
        loss_ratio = loss_region.sum() / total_px * 100
        fig.suptitle(
            f'样本 {idx}  |  面波占比: {sw_ratio:.2f}%  |  '
            f'损失区域: {loss_ratio:.2f}%  |  高斯范围: [-{self.mask_k}, {self.mask_k}]  |  '
            f'盲扇模式: {mode_label}',
            fontsize=13, y=1.01
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
        gather_dir           = r"原始炮集",
        mask_dir             = r"masks",
        total                = 192,
        prefix_gather        = "real_data_gather",
        prefix_mask          = "mask_",
        max_time_samples     = None,
        non_sw_density_range = (0.08, 0.12),
        mask_k               = 0.5,
        enable_augmentation  = True,
        loss_on_full_non_sw  = True,
        normalize            = True,
        blind_spot_mode      = 'full_line',  # 'half_line' 或 'full_line'
    )
    dataset.shot_center = 85

    print(f"数据集大小: {len(dataset)}")

    fig = dataset.visualize_sample(69)
    plt.show()