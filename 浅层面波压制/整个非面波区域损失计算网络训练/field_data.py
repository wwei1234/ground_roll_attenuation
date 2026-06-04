import numpy as np
import matplotlib.pyplot as plt
import segyio
from torch.utils.data import Dataset
from matplotlib import rcParams # 导入 Matplotlib 配置

# ==============================================================================
# 实用工具函数和类
# ==============================================================================

class SeismicEnergyAnalyzer:
    """地震数据能量片段分析器，用于识别面波区域"""
    
    def __init__(self, window_width=0.02, time_shift=0.01):
        self.window_width = window_width
        self.time_shift = time_shift
        
    def gaussian_window(self, t, center, alpha):
        """计算高斯窗函数值"""
        X = np.sqrt(alpha / np.pi)
        # 注意：这里的 alpha^2 可能是笔误，物理上通常是 alpha
        # 为了兼容原代码逻辑，保留原写法，但更标准的写法是 np.exp(-(t - center)**2 / (2 * sigma**2))
        window = X * np.exp(-alpha**2 * (t - center)**2)
        return window
    
    def compute_segment_energy(self, seismic_trace, dt):
        """计算逐段能量（使用高斯窗）"""
        n_samples = len(seismic_trace)
        t = np.arange(n_samples) * dt
        alpha = 1.0 / self.window_width
        n_windows = int((t[-1] - t[0]) / self.time_shift) + 1
        time_centers = np.arange(n_windows) * self.time_shift + t[0]
        
        energies = []
        for center in time_centers:
            window = self.gaussian_window(t, center, alpha)
            segment = seismic_trace * window
            # 使用平均绝对振幅作为能量代理
            avg_amplitude = np.mean(np.abs(segment))
            energies.append(avg_amplitude)
        
        return time_centers, np.array(energies)
    
    def identify_surface_wave(self, energies, threshold=0.001):
        """简单阈值判断面波"""
        return energies > threshold
    
    def create_trace_mask(self, seismic_trace, dt, threshold=0.001):
        """为单道生成面波掩码（二值）"""
        n_samples = len(seismic_trace)
        time_centers, energies = self.compute_segment_energy(seismic_trace, dt)
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        mask = np.zeros(n_samples, dtype=int)
        
        for i, center in enumerate(time_centers):
            if surface_wave_mask[i]:
                # 设定窗口区域（简单矩形窗）
                t_start = center - self.window_width / 2
                t_end = center + self.window_width / 2
                # 转换为采样点索引
                idx_start = max(0, int(t_start / dt))
                idx_end = min(n_samples, int(t_end / dt))
                mask[idx_start:idx_end] = 1
        
        return mask
    
    def batch_identify_surface_wave(self, shot_gather, dt, threshold=0.001):
        """为炮集生成面波掩码（批量）"""
        n_samples, n_traces = shot_gather.shape
        mask_gather = np.zeros_like(shot_gather, dtype=int)
        
        for i in range(n_traces):
            trace = shot_gather[:, i]
            mask = self.create_trace_mask(trace, dt, threshold)
            mask_gather[:, i] = mask
        
        return mask_gather

def normalize_traces_per_trace(shot_gather):
    """逐道归一化"""
    normalized_gather = np.copy(shot_gather)
    # 确保数据是浮点型以便进行除法操作
    if normalized_gather.dtype != np.float32 and normalized_gather.dtype != np.float64:
        normalized_gather = normalized_gather.astype(np.float32)
        
    n_samples, n_traces = normalized_gather.shape
    
    for i in range(n_traces):
        trace = normalized_gather[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized_gather[:, i] = trace / max_amp
    
    return normalized_gather


def remove_direct_wave(data, center_trace=64.5, velocity=800, dt=0.00025, 
                       mute_time0=0.010, dx=2, taper=20):
    """
    去除直达波（走时线切除和衰减）
    center_trace: 炮点道号 (可能带小数)
    """
    n_time, n_trace = data.shape
    muted_data = data.copy()
    mask = np.ones_like(data)

    for tr in range(n_trace):
        # 计算偏移距
        offset = abs(tr - center_trace) * dx
        # 计算直达波起始走时 (t = t0 + offset / velocity)
        t0 = mute_time0 + offset / velocity
        t0_idx = int(t0 / dt)

        if t0_idx < n_time:
            # 1. 绝对切除（直达波以上区域）
            mask[:t0_idx, tr] = 0
            
            # 2. 锥形衰减 (Taper)
            taper_end = min(t0_idx + taper, n_time)
            for t in range(t0_idx, taper_end):
                # 线性衰减因子从 1 递减到 0
                mask[t, tr] = 1 - (t - t0_idx) / taper

    muted_data *= mask
    return muted_data

# ==============================================================================
# 数据集类 SeismicMaskDataset_FIELD (已修正/优化)
# ==============================================================================

class SeismicMaskDataset_FIELD(Dataset):
    """
    地震数据掩码数据集生成器（支持循环炮点位置）
    """
    
    def __init__(self, segy_path, 
                 # 炮点位置配置
                 shot_center_mode='fixed',
                 shot_center_fixed=24,
                 shot_center_start=12,
                 shot_center_end=35,
                 # 其他参数
                 dt=0.00025,
                 non_sw_density_range=(5, 15),
                 shotnum=700, max_time_samples=1200,
                 # 面波识别参数
                 window_width=0.01, time_shift=0.002, sw_threshold=0.08,
                 # 直达波去除参数
                 remove_direct=True, direct_velocity=800, 
                 mute_time0=0.010, dx=2, taper=20,
                 # 数据增强参数
                 enable_augmentation=True,
                 # 掩码参数
                 mask_k=1.0,
                 # 损失计算范围参数
                 loss_on_full_non_sw=True,
                 include_above_direct=False):
        
        # 炮点位置配置
        self.shot_center_mode = shot_center_mode
        self.shot_center_fixed = shot_center_fixed
        self.shot_center_start = shot_center_start
        self.shot_center_end = shot_center_end
        self.shot_center_cycle_length = shot_center_end - shot_center_start + 1
        
        self.dt = dt
        self.non_sw_density_range = non_sw_density_range
        self.max_time_samples = max_time_samples
        self.remove_direct = remove_direct
        self.direct_velocity = direct_velocity
        self.mute_time0 = mute_time0
        self.taper = taper
        self.dx = dx
        self.enable_augmentation = enable_augmentation
        self.mask_k = mask_k
        
        # 损失计算范围参数
        self.loss_on_full_non_sw = loss_on_full_non_sw
        self.include_above_direct = include_above_direct
        
        # 创建面波分析器
        self.analyzer = SeismicEnergyAnalyzer(
            window_width=window_width, 
            time_shift=time_shift
        )
        self.sw_threshold = sw_threshold
        
        # 读取SEGY数据 (修正: 确保读取为 float32)
        print("=" * 70)
        print("正在初始化地震数据掩码数据集（统一高斯掩码策略）...")
        print("=" * 70)
        self.seismic_data = self.read_segy(segy_path, shotnum)
        
        # 只保留前max_time_samples个采样点
        self.seismic_data = self.seismic_data[:, :max_time_samples, :]
        
        self.original_shot_num, self.time_samples, self.trace_num = self.seismic_data.shape
        
        print(f"\n数据集信息:")
        print(f"  原始炮数: {self.original_shot_num}")
        print(f"  时间采样点: {self.time_samples}")
        print(f"  道数: {self.trace_num}")
        
        # 炮点位置信息
        if self.shot_center_mode == 'cyclic':
            print(f"  炮点位置模式: 循环")
            print(f"    循环范围: 第{self.shot_center_start}道 ~ 第{self.shot_center_end}道")
            print(f"    循环周期: {self.shot_center_cycle_length}炮")
        else:
            print(f"  炮点位置模式: 固定")
            print(f"    固定位置: 第{self.shot_center_fixed}道")
        
        # ... (省略打印其他策略信息，与原代码相同) ...
        print(f"\n统一高斯掩码策略:")
        print(f"  【输入】:")
        print(f"    - 面波区域: 全部替换为高斯分布掩码 ([-{self.mask_k}, {self.mask_k}])")
        print(f"    - 非面波区域: 原始数据 + 部分位置替换为高斯分布掩码 ([-{self.mask_k}, {self.mask_k}])")
        print(f"  【标签】:")
        print(f"    - 面波区域: 与输入相同的高斯掩码")
        print(f"    - 非面波区域: 原始数据（无掩码）")
        print(f"  【损失计算】:")
        if self.loss_on_full_non_sw:
            print(f"    - 计算范围: 全部非面波区域")
        else:
            print(f"    - 计算范围: 仅非面波掩码位置")
        if self.include_above_direct:
            print(f"    - 直达波区域: 包含直达波以上区域")
        else:
            print(f"    - 直达波区域: 仅直达波以下区域")
        
        if self.enable_augmentation:
            print(f"\n数据增强: 启用 (左右翻转)")
            print(f"  增强后总样本数: {self.original_shot_num * 2}")
        else:
            print(f"\n数据增强: 未启用")

        
        # 预处理所有炮集并识别面波区域
        print(f"\n开始预处理数据并识别面波区域...")
        self.processed_data = []
        self.surface_wave_masks = []
        self.shot_centers = []
        
        # 处理原始数据
        for shot_idx in range(self.original_shot_num):
            current_shot_center = self.get_shot_center(shot_idx)
            self.shot_centers.append(current_shot_center)
            
            # 归一化
            shot_gather = normalize_traces_per_trace(self.seismic_data[shot_idx])
            
            # 去除直达波
            if self.remove_direct:
                shot_gather = remove_direct_wave(
                    shot_gather,
                    center_trace=current_shot_center,
                    velocity=self.direct_velocity,
                    dt=self.dt,
                    mute_time0=self.mute_time0,
                    dx=self.dx,
                    taper=self.taper
                )
            
            # 识别面波区域
            sw_mask = self.analyzer.batch_identify_surface_wave(
                shot_gather, 
                self.dt, 
                self.sw_threshold
            )
            
            self.processed_data.append(shot_gather)
            self.surface_wave_masks.append(sw_mask)
            
            if (shot_idx + 1) % 50 == 0 or shot_idx < 5:
                print(f"  已处理原始数据 {shot_idx + 1}/{self.original_shot_num} 炮 "
                      f"(炮点位置: 第{current_shot_center}道)")
        
        # 数据增强
        if self.enable_augmentation:
            print(f"\n开始生成翻转数据...")
            for shot_idx in range(self.original_shot_num):
                flipped_gather = np.fliplr(self.processed_data[shot_idx])
                flipped_sw_mask = np.fliplr(self.surface_wave_masks[shot_idx])
                
                original_shot_center = self.shot_centers[shot_idx]
                # 翻转后的炮点位置
                flipped_shot_center = self.trace_num - 1 - original_shot_center
                
                self.processed_data.append(flipped_gather)
                self.surface_wave_masks.append(flipped_sw_mask)
                self.shot_centers.append(flipped_shot_center)
                
                if (shot_idx + 1) % 50 == 0 or shot_idx < 5:
                    print(f"  已生成翻转数据 {shot_idx + 1}/{self.original_shot_num} 炮 "
                          f"(炮点: {original_shot_center}→{flipped_shot_center})")
        
        self.processed_data = np.array(self.processed_data)
        self.surface_wave_masks = np.array(self.surface_wave_masks)
        self.shot_centers = np.array(self.shot_centers)
        
        self.shot_num = len(self.processed_data)
        
        print(f"\n数据预处理完成！")
        print(f"  最终数据集大小: {self.shot_num} 炮")
        if self.enable_augmentation:
            print(f"  (原始: {self.original_shot_num} + 翻转: {self.original_shot_num})")
        print("=" * 70)
    
    def get_shot_center(self, shot_idx):
        """根据炮号获取原始炮点位置"""
        if self.shot_center_mode == 'cyclic':
            # 确保 shot_idx 不超过 original_shot_num
            idx_in_cycle = shot_idx % self.shot_center_cycle_length
            shot_center = self.shot_center_start + idx_in_cycle
        else:
            shot_center = self.shot_center_fixed
        
        return shot_center
    
    def read_segy(self, data_dir, shotnum=0):
        """
        读取SEGY数据。
        修正点：强制将读取的数据转换为 np.float32。
        """
        with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)
            if shotnum:
                shot_num = shotnum 
            else:
                shot_num = len(set(sourceX))
            len_shot = trace_num // shot_num
            time = f.trace[0].shape[0]
            print(f'读取SEGY数据: {shot_num}炮, 每炮{len_shot}道, {time}个时间样点')
            
            # 确保 data 数组是 float32 类型
            data = np.zeros((shot_num, time, len_shot), dtype=np.float32)
            
            for j in range(shot_num):
                # 读取时强制转换为 float32
                trace_data = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
                data[j, :, :] = trace_data.astype(np.float32)
                
            return data
    
    def is_flipped_sample(self, idx):
        """判断索引是否对应翻转样本"""
        if not self.enable_augmentation:
            return False
        return idx >= self.original_shot_num
    
    def get_shot_center_for_idx(self, idx):
        """获取对应索引的炮点位置"""
        return self.shot_centers[idx]
    
    def get_line_pixels(self, point_trace, point_time, shot_center):
        """获取从活跃像素沿炮点连线方向反向（向下）延伸的所有像素位置"""
        pixels = []
        t0, x0 = 0, shot_center
        t1, x1 = point_time, point_trace

        dt = t1 - t0
        dx = x1 - x0

        if dt == 0 and dx == 0:
            return [(t1, x1)]

        length = np.hypot(dx, dt)
        dx /= length
        dt /= length

        if dt <= 0:
            dx = -dx
            dt = abs(dt)

        x, t = x1, t1

        while 0 <= int(x) < self.trace_num and 0 <= int(t) < self.time_samples:
            pixels.append((int(round(t)), int(round(x))))
            x += dx
            t += dt

        return pixels

    def get_direct_wave_mask(self, center_trace, velocity, mute_time0, dx, dt):
        """计算直达波区域掩码（直达波以下区域为 True）"""
        direct_mask = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        
        for tr in range(self.trace_num):
            offset = abs(tr - center_trace) * dx
            t0 = mute_time0 + offset / velocity
            t0_idx = int(t0 / dt)
            
            if t0_idx < self.time_samples:
                # 直达波以下区域为 True
                direct_mask[t0_idx:, tr] = True
        
        return direct_mask
    
    def generate_mask(self, shot_idx):
        """为指定炮生成掩码（统一高斯掩码策略）"""
        original_data = self.processed_data[shot_idx].copy()
        sw_mask = self.surface_wave_masks[shot_idx]
        current_shot_center = self.get_shot_center_for_idx(shot_idx)
        
        direct_wave_mask = self.get_direct_wave_mask(
            current_shot_center, self.direct_velocity, self.mute_time0, self.dx, self.dt
        )
        
        # 根据参数决定有效区域
        if self.include_above_direct:
            valid_region = np.ones((self.time_samples, self.trace_num), dtype=bool)
        else:
            valid_region = direct_wave_mask
        
        valid_sw_mask = sw_mask.astype(bool) & valid_region
        valid_non_sw_mask = (~sw_mask.astype(bool)) & valid_region
        
        # 生成高斯掩码
        gaussian_mask = np.random.normal(0, self.mask_k / 3.0, size=(self.time_samples, self.trace_num))
        # 强制转换为原始数据类型 (np.float32)
        gaussian_mask = np.clip(gaussian_mask, -self.mask_k, self.mask_k).astype(original_data.dtype)
        
        # 生成非面波掩码位置
        non_sw_mask_positions = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        
        valid_traces = np.where(np.any(valid_region, axis=0))[0]
        
        if len(valid_traces) > 0:
            for trace_idx in valid_traces:
                non_sw_pixels_in_trace = np.where(valid_non_sw_mask[:, trace_idx])[0]
                
                if len(non_sw_pixels_in_trace) > 0:
                    non_sw_density = np.random.uniform(*self.non_sw_density_range)
                    n_active_non_sw = max(1, int(len(non_sw_pixels_in_trace) * non_sw_density / 100))
                    n_active_non_sw = min(n_active_non_sw, len(non_sw_pixels_in_trace))
                    
                    selected_times = np.random.choice(
                        non_sw_pixels_in_trace, 
                        size=n_active_non_sw, 
                        replace=False
                    )
                    
                    for time_idx in selected_times:
                        # 使用修正后的 get_line_pixels
                        line_pixels = self.get_line_pixels(trace_idx, time_idx, current_shot_center)
                        
                        for t, x in line_pixels:
                            if (0 <= t < self.time_samples and 
                                0 <= x < self.trace_num and
                                valid_region[t, x] and
                                not sw_mask[t, x]):
                                non_sw_mask_positions[t, x] = True
        
        # 构建输入和标签
        input_data = original_data.copy()
        input_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        input_data[non_sw_mask_positions] = gaussian_mask[non_sw_mask_positions]
        
        label_data = original_data.copy()
        label_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        
        non_sw_region = valid_non_sw_mask
        
        if self.loss_on_full_non_sw:
            loss_region = non_sw_region
        else:
            loss_region = non_sw_mask_positions
        
        return input_data, label_data, non_sw_mask_positions, non_sw_region, loss_region, sw_mask, original_data
    
    def __len__(self):
        return self.shot_num
    
    def __getitem__(self, idx):
        input_data, label_data, non_sw_mask_positions, non_sw_region, loss_region, sw_mask, original_data = self.generate_mask(idx)
        
        return {
            'input_data': input_data,
            'label_data': label_data,
            'non_sw_mask_positions': non_sw_mask_positions,
            'non_sw_region': non_sw_region,
            'loss_region': loss_region,
            'surface_wave_mask': sw_mask,
            'original_data': original_data,
            'shot_idx': idx,
            'is_flipped': self.is_flipped_sample(idx),
            'original_shot_idx': idx if not self.is_flipped_sample(idx) else idx - self.original_shot_num
        }
    
    def visualize_sample(self, idx, figsize=(18, 10)):
        """
        修正后的可视化函数：只绘制 6 张核心图。
        """
        sample = self.__getitem__(idx)
        
        input_data = sample['input_data']
        label_data = sample['label_data']
        original_data = sample['original_data']
        non_sw_mask_positions = sample['non_sw_mask_positions']
        loss_region = sample['loss_region']
        sw_mask = sample['surface_wave_mask']
        is_flipped = sample['is_flipped']
        original_shot_idx = sample['original_shot_idx']
        
        current_shot_center = self.get_shot_center_for_idx(idx)
        
        # 计算有效区域
        if self.include_above_direct:
            valid_region = np.ones((self.time_samples, self.trace_num), dtype=bool)
        else:
            valid_region = self.get_direct_wave_mask(
                current_shot_center, self.direct_velocity, self.mute_time0, self.dx, self.dt
            )
        
        # 准备用于显示的有效面波掩码
        sw_mask_display = (sw_mask.astype(bool) & valid_region).astype(int)
            
        fig, axes = plt.subplots(2, 3, figsize=figsize) 
        axes = axes.flatten() 
        
        # 计算显示范围
        vmax = np.percentile(np.abs(original_data), 99)
        
        # --- 子图 1: 原始数据 ---
        im0 = axes[0].imshow(original_data, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear')
        title_prefix = f'翻转样本 - 原始第{original_shot_idx}炮' if is_flipped else f'原始样本 - 第{idx}炮'
        axes[0].set_title(f'1. 原始数据 - {title_prefix}', fontsize=12, fontweight='bold')
        axes[0].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2, label='炮点')
        axes[0].set_xlabel('道号'); axes[0].set_ylabel('时间采样点')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # --- 子图 2: 面波区域识别 ---
        im1 = axes[1].imshow(sw_mask_display, aspect='auto', cmap='Reds',
                             vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title(f'2. 面波区域识别 (SW Mask)', fontsize=12, color='red')
        axes[1].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[1].set_xlabel('道号'); axes[1].set_ylabel('时间采样点')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # --- 子图 3: 非面波区域高斯掩码位置显示 ---
        im2 = axes[2].imshow(non_sw_mask_positions.astype(int), aspect='auto', 
                             cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_title(f'3. 非面波高斯掩码位置 (Non-SW Mask Positions)', fontsize=12, color='blue')
        axes[2].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[2].set_xlabel('道号'); axes[2].set_ylabel('时间采样点')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # --- 子图 4: 网络输入 ---
        im3 = axes[3].imshow(input_data, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[3].set_title(f'4. 网络输入 (Input Data)', fontsize=12, fontweight='bold', color='blue')
        axes[3].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[3].set_xlabel('道号'); axes[3].set_ylabel('时间采样点')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        # --- 子图 5: 网络标签 ---
        im4 = axes[4].imshow(label_data, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[4].set_title(f'5. 网络标签 (Label Data)', fontsize=12, fontweight='bold', color='green')
        axes[4].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[4].set_xlabel('道号'); axes[4].set_ylabel('时间采样点')
        plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        
        # --- 子图 6: 损失函数计算位置图 ---
        im5 = axes[5].imshow(loss_region.astype(int), aspect='auto', 
                             cmap='Oranges', vmin=0, vmax=1, interpolation='nearest')
        loss_title = '6. 损失计算区域（全部非面波）' if self.loss_on_full_non_sw else '6. 损失计算区域（仅掩码位置）'
        axes[5].set_title(loss_title, fontsize=12, fontweight='bold', color='darkorange')
        axes[5].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[5].set_xlabel('道号'); axes[5].set_ylabel('时间采样点')
        plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

        # 统计信息作为主标题
        valid_area = valid_region.sum()
        sw_ratio = sw_mask_display.sum() / valid_area * 100 if valid_area > 0 else 0
        loss_ratio = np.sum(loss_region) / valid_area * 100 if valid_area > 0 else 0
        
        flip_status = " [翻转样本]" if is_flipped else " [原始样本]"
        direct_status = "全图" if self.include_above_direct else "直达波以下"
        loss_mode = "全部非面波" if self.loss_on_full_non_sw else "仅掩码位置"
        info_text = (f'样本 {idx}{flip_status} | 炮点: {current_shot_center} | 面波区域: {sw_ratio:.2f}% | '
                      f'损失区域: {loss_ratio:.2f}% ({loss_mode}) | '
                      f'有效范围: {direct_status} | 高斯范围: [-{self.mask_k}, {self.mask_k}]')
        
        fig.suptitle(info_text, fontsize=14, y=1.02)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 配置Matplotlib以支持中文显示
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # 假设 SEGY 文件路径
    # 请注意：要运行此代码，您需要将 'segy_file_path' 更改为有效的 SEGY 文件路径。
    # 如果您没有文件，可以尝试注释掉 try/except 块，只保留类定义。
    segy_file_path = r"D:\桌面\面波压制\data\modified_SN03.sgy"
    # 创建数据集
    dataset = SeismicMaskDataset_FIELD(
        segy_path=segy_file_path,
        shot_center_mode='cyclic', # 循环炮点位置
        shot_center_fixed=24,
        shot_center_start=12,
        shot_center_end=35,
        dt=0.00025,
        non_sw_density_range=(0, 0.05), # 非面波区域掩码的稀疏度极低
        shotnum=168,
        max_time_samples=1200,
        window_width=0.01,
        time_shift=0.002,
        sw_threshold=0.08,
        remove_direct=True,
        direct_velocity=1200,
        mute_time0=0.010,
        dx=2,
        taper=20,
        enable_augmentation=False,
        mask_k=0.4,
        loss_on_full_non_sw=True,      # True: 全部非面波区域计算损失
        include_above_direct=False     # False: 仅直达波以下区域有效
    )
    
    print(f"\n数据集总样本数: {len(dataset)}")
    # 可视化样本
    sample_index_to_visualize = 13  # 示例样本
    print(f"\n正在生成样本 {sample_index_to_visualize} 的简化可视化...")
    fig1 = dataset.visualize_sample(sample_index_to_visualize)
    plt.show()
    
    
