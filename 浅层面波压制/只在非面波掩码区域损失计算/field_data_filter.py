import numpy as np
import matplotlib.pyplot as plt
import segyio
from torch.utils.data import Dataset
from matplotlib import rcParams
from scipy import signal

class SeismicEnergyAnalyzer:
    """地震数据能量片段分析器"""
    
    def __init__(self, window_width=0.02, time_shift=0.01):
        self.window_width = window_width
        self.time_shift = time_shift
        
    def gaussian_window(self, t, center, alpha):
        X = np.sqrt(alpha / np.pi)
        window = X * np.exp(-alpha**2 * (t - center)**2)
        return window
    
    def compute_segment_energy(self, seismic_trace, dt):
        n_samples = len(seismic_trace)
        t = np.arange(n_samples) * dt
        alpha = 1.0 / self.window_width
        n_windows = int((t[-1] - t[0]) / self.time_shift) + 1
        time_centers = np.arange(n_windows) * self.time_shift + t[0]
        
        energies = []
        for center in time_centers:
            window = self.gaussian_window(t, center, alpha)
            segment = seismic_trace * window
            avg_amplitude = np.mean(np.abs(segment))
            energies.append(avg_amplitude)
        
        return time_centers, np.array(energies)
    
    def identify_surface_wave(self, energies, threshold=0.001):
        return energies > threshold
    
    def create_trace_mask(self, seismic_trace, dt, threshold=0.001):
        n_samples = len(seismic_trace)
        t = np.arange(n_samples) * dt
        time_centers, energies = self.compute_segment_energy(seismic_trace, dt)
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        mask = np.zeros(n_samples, dtype=int)
        
        for i, center in enumerate(time_centers):
            if surface_wave_mask[i]:
                t_start = center - self.window_width / 2
                t_end = center + self.window_width / 2
                idx_start = max(0, int(t_start / dt))
                idx_end = min(n_samples, int(t_end / dt))
                mask[idx_start:idx_end] = 1
        
        return mask
    
    def batch_identify_surface_wave(self, shot_gather, dt, threshold=0.001):
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
    n_samples, n_traces = normalized_gather.shape
    
    for i in range(n_traces):
        trace = normalized_gather[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized_gather[:, i] = trace / max_amp
    
    return normalized_gather


def remove_direct_wave(data, center_trace=64.5, velocity=800, dt=0.00025, 
                       mute_time0=0.010, dx=2, taper=20):
    """去除直达波"""
    n_time, n_trace = data.shape
    muted_data = data.copy()
    mask = np.ones_like(data)

    for tr in range(n_trace):
        offset = abs(tr - center_trace)*dx
        t0 = mute_time0 + offset / velocity
        t0_idx = int(t0/dt)

        if t0_idx < n_time:
            mask[:t0_idx, tr] = 0
            taper_end = min(t0_idx + taper, n_time)
            for t in range(t0_idx, taper_end):
                mask[t, tr] = 1 - (t - t0_idx) / taper

    muted_data *= mask
    return muted_data


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    带通滤波器
    
    参数:
        data: 输入数据，shape为 (时间采样点, 道数)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        fs: 采样率 (Hz)
        order: 滤波器阶数
    
    返回:
        filtered_data: 滤波后的数据
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = signal.sosfiltfilt(sos, data[:, i])
    
    return filtered_data


class SeismicMaskDataset_FIELD(Dataset):
    """
    地震数据掩码数据集生成器（支持循环炮点位置、带通滤波、基于速度的阈值）
    """
    
    def __init__(self, segy_path, 
                 shot_center_mode='fixed', 
                 shot_center_fixed=24, 
                 shot_center_start=12, 
                 shot_center_end=35, 
                 dt=0.00025,
                 non_sw_density_range=(5, 15),
                 shotnum=700, max_time_samples=1200,
                 window_width=0.01, time_shift=0.002, sw_threshold=0.08,
                 remove_direct=True, direct_velocity=800, 
                 mute_time0=0.010, dx=2, taper=20,
                 enable_augmentation=True,
                 mask_k=1.0,
                 # 带通滤波参数
                 enable_bandpass=False,
                 bandpass_low=20,
                 bandpass_high=50,
                 sample_rate=4000,
                 bandpass_order=4,
                 # 基于速度的阈值参数
                 velocity_regions=None,
                 apply_velocity_constraint=False,
                 sw_min_velocity=200,
                 sw_max_velocity=800):
        
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
        
        # 带通滤波参数
        self.enable_bandpass = enable_bandpass
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.sample_rate = sample_rate
        self.bandpass_order = bandpass_order
        
        # 基于速度的阈值参数
        self.velocity_regions = velocity_regions if velocity_regions else []
        self.apply_velocity_constraint = apply_velocity_constraint
        self.sw_min_velocity = sw_min_velocity
        self.sw_max_velocity = sw_max_velocity
        
        # 创建面波分析器
        self.analyzer = SeismicEnergyAnalyzer(
            window_width=window_width, 
            time_shift=time_shift
        )
        self.sw_threshold = sw_threshold
        
        # 读取SEGY数据
        print("=" * 70)
        print("正在初始化地震数据掩码数据集...")
        print("=" * 70)
        self.seismic_data = self.read_segy(segy_path, shotnum)
        self.seismic_data = self.seismic_data[:, :max_time_samples, :]
        
        self.original_shot_num, self.time_samples, self.trace_num = self.seismic_data.shape
        
        print(f"\n数据集信息:")
        print(f"  原始炮数: {self.original_shot_num}")
        print(f"  时间采样点: {self.time_samples}")
        print(f"  道数: {self.trace_num}")
        
        if self.shot_center_mode == 'cyclic':
            print(f"  炮点位置模式: 循环 (范围: {self.shot_center_start}~{self.shot_center_end})")
        else:
            print(f"  炮点位置模式: 固定 (位置: {self.shot_center_fixed})")
        
        if self.enable_bandpass:
            print(f"  带通滤波: 启用 ({self.bandpass_low}-{self.bandpass_high} Hz)")
        else:
            print(f"  带通滤波: 未启用")
        
        if self.velocity_regions:
            print(f"  基于速度的阈值: 启用 ({len(self.velocity_regions)} 个区域)")
        else:
            print(f"  基于速度的阈值: 未启用 (使用默认阈值 {self.sw_threshold})")
        
        if self.enable_augmentation:
            print(f"  数据增强: 启用 (左右翻转) -> 总样本数: {self.original_shot_num * 2}")
        else:
            print(f"  数据增强: 未启用")
        
        # 预处理所有炮集并识别面波区域
        print(f"\n开始预处理数据并识别面波区域...")
        self.processed_data = []
        self.surface_wave_masks = []
        self.shot_centers = [] 
        
        for shot_idx in range(self.original_shot_num):
            current_shot_center = self.get_shot_center(shot_idx)
            self.shot_centers.append(current_shot_center)
            
            shot_gather = normalize_traces_per_trace(self.seismic_data[shot_idx])
            
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
            
            # 带通滤波（在去直达波之后）
            if self.enable_bandpass:
                shot_gather = bandpass_filter(
                    shot_gather,
                    self.bandpass_low,
                    self.bandpass_high,
                    self.sample_rate,
                    self.bandpass_order
                )
            
            # 基于速度的面波识别
            if self.velocity_regions:
                sw_mask, _ = self.identify_surface_wave_velocity_based(
                    shot_gather, current_shot_center
                )
            else:
                sw_mask = self.analyzer.batch_identify_surface_wave(
                    shot_gather, self.dt, self.sw_threshold
                )
            
            self.processed_data.append(shot_gather)
            self.surface_wave_masks.append(sw_mask)
            
            if (shot_idx + 1) % 50 == 0 or shot_idx < 5:
                print(f"  已处理原始数据 {shot_idx + 1}/{self.original_shot_num} 炮 (炮点位置: 第{current_shot_center}道)")
            
        # 数据增强
        if self.enable_augmentation:
            print(f"\n开始生成翻转数据...")
            for shot_idx in range(self.original_shot_num):
                flipped_gather = np.fliplr(self.processed_data[shot_idx])
                flipped_sw_mask = np.fliplr(self.surface_wave_masks[shot_idx])
                
                original_shot_center = self.shot_centers[shot_idx]
                flipped_shot_center = self.trace_num - 1 - original_shot_center
                
                self.processed_data.append(flipped_gather)
                self.surface_wave_masks.append(flipped_sw_mask)
                self.shot_centers.append(flipped_shot_center)
                
                if (shot_idx + 1) % 50 == 0 or shot_idx < 5:
                    print(f"  已生成翻转数据 {shot_idx + 1}/{self.original_shot_num} 炮")
            
        self.processed_data = np.array(self.processed_data)
        self.surface_wave_masks = np.array(self.surface_wave_masks)
        self.shot_centers = np.array(self.shot_centers)
        
        self.shot_num = len(self.processed_data)
        
        print(f"\n数据预处理完成！最终数据集大小: {self.shot_num} 炮")
        print("=" * 70)
    
    def get_shot_center(self, shot_idx):
        """根据原始炮号获取炮点位置"""
        if self.shot_center_mode == 'cyclic':
            shot_center = self.shot_center_start + (shot_idx % self.shot_center_cycle_length)
        else:
            shot_center = self.shot_center_fixed
        return shot_center
    
    def read_segy(self, data_dir, shotnum=0):
        """读取SEGY数据"""
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
            
            data = np.zeros((shot_num, time, len_shot), dtype=np.float32)
            
            for j in range(shot_num):
                trace_data = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
                data[j, :, :] = trace_data.astype(np.float32)
                
            return data
    
    def get_velocity_at_point(self, time_idx, trace_idx, shot_center):
        """计算某点的视速度"""
        offset = (trace_idx - shot_center) * self.dx
        time = time_idx * self.dt
        
        if time <= 0 or offset == 0:
            return None
        
        velocity = offset / time
        return velocity
    
    def get_direct_wave_mask(self, center_trace):
        """计算直达波区域掩码"""
        direct_mask = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        
        for tr in range(self.trace_num):
            offset = abs(tr - center_trace) * self.dx
            t0 = self.mute_time0 + offset / self.direct_velocity
            t0_idx = int(t0 / self.dt)
            
            if t0_idx < self.time_samples:
                direct_mask[t0_idx:, tr] = True
        
        return direct_mask
    
    def apply_velocity_constraint_func(self, sw_mask, shot_center):
        """应用速度约束"""
        time_samples, trace_num = sw_mask.shape
        velocity_filtered_mask = np.zeros_like(sw_mask, dtype=bool)
        
        for tr in range(trace_num):
            offset = abs(tr - shot_center) * self.dx
            
            if offset == 0:
                velocity_filtered_mask[:, tr] = sw_mask[:, tr]
                continue
            
            t_min = offset / self.sw_max_velocity
            t_max = offset / self.sw_min_velocity
            
            idx_min = max(0, int(t_min / self.dt))
            idx_max = min(time_samples - 1, int(t_max / self.dt))
            
            if idx_min < time_samples:
                velocity_window = np.zeros(time_samples, dtype=bool)
                velocity_window[idx_min:idx_max+1] = True
                velocity_filtered_mask[:, tr] = sw_mask[:, tr] & velocity_window
        
        return velocity_filtered_mask
    
    def identify_surface_wave_velocity_based(self, shot_gather, shot_center):
        """
        基于速度的自定义阈值面波识别
        
        velocity_regions格式:
        [
            {
                'velocity_range': (min_vel, max_vel),
                'threshold': 0.03
            },
            ...
        ]
        """
        time_samples, trace_num = shot_gather.shape
        sw_mask = np.zeros((time_samples, trace_num), dtype=bool)
        
        # 创建速度阈值映射表
        threshold_map = np.full((time_samples, trace_num), self.sw_threshold)
        
        if self.velocity_regions:
            for t_idx in range(time_samples):
                for tr_idx in range(trace_num):
                    velocity = self.get_velocity_at_point(t_idx, tr_idx, shot_center)
                    
                    if velocity is None:
                        continue
                    
                    for region in self.velocity_regions:
                        v_min, v_max = region['velocity_range']
                        threshold = region['threshold']
                        
                        if v_min > v_max:
                            v_min, v_max = v_max, v_min
                        
                        if v_min <= velocity <= v_max:
                            threshold_map[t_idx, tr_idx] = threshold
                            break
        
        # 对每道应用相应的阈值
        for tr in range(trace_num):
            trace = shot_gather[:, tr]
            time_centers, energies = self.analyzer.compute_segment_energy(trace, self.dt)
            
            mask = np.zeros(time_samples, dtype=bool)
            
            for i, center in enumerate(time_centers):
                t_idx = int(center / self.dt)
                if t_idx >= time_samples:
                    continue
                
                threshold = threshold_map[t_idx, tr]
                
                if energies[i] > threshold:
                    t_start = center - self.analyzer.window_width / 2
                    t_end = center + self.analyzer.window_width / 2
                    idx_start = max(0, int(t_start / self.dt))
                    idx_end = min(time_samples, int(t_end / self.dt))
                    mask[idx_start:idx_end] = True
            
            sw_mask[:, tr] = mask
        
        # 应用速度约束
        if self.apply_velocity_constraint:
            sw_mask = self.apply_velocity_constraint_func(sw_mask, shot_center)
        
        # 只保留直达波以下的面波
        direct_wave_mask = self.get_direct_wave_mask(shot_center)
        valid_sw_mask = sw_mask & direct_wave_mask
        
        return valid_sw_mask.astype(int), threshold_map
    
    def is_flipped_sample(self, idx):
        """判断索引是否对应翻转样本"""
        if not self.enable_augmentation:
            return False
        return idx >= self.original_shot_num
    
    def get_shot_center_for_idx(self, idx):
        """获取对应索引的炮点位置"""
        return self.shot_centers[idx]
    
    def get_line_pixels(self, point_trace, point_time, shot_center):
        """获取从活跃像素沿炮点连线方向反向延伸的所有像素位置"""
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
    
    def generate_mask(self, shot_idx):
        """为指定炮生成掩码"""
        original_data = self.processed_data[shot_idx].copy()
        sw_mask = self.surface_wave_masks[shot_idx]
        
        current_shot_center = self.get_shot_center_for_idx(shot_idx)
        direct_wave_mask = self.get_direct_wave_mask(current_shot_center)
        
        valid_sw_mask = (sw_mask & direct_wave_mask).astype(bool)
        valid_non_sw_mask = (~sw_mask.astype(bool)) & direct_wave_mask
        
        gaussian_mask = np.random.normal(0, self.mask_k / 3.0, 
                                         size=(self.time_samples, self.trace_num)).astype(original_data.dtype)
        gaussian_mask = np.clip(gaussian_mask, -self.mask_k, self.mask_k)
        
        non_sw_mask_positions = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        
        valid_traces = np.where(np.any(direct_wave_mask, axis=0))[0]
        
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
                        line_pixels = self.get_line_pixels(trace_idx, time_idx, current_shot_center)
                        
                        for t, x in line_pixels:
                            if (0 <= t < self.time_samples and 
                                0 <= x < self.trace_num and
                                direct_wave_mask[t, x] and
                                not sw_mask[t, x]):
                                non_sw_mask_positions[t, x] = True
        
        input_data = original_data.copy()
        input_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        input_data[non_sw_mask_positions] = gaussian_mask[non_sw_mask_positions]
        
        label_data = original_data.copy()
        label_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        
        return input_data, label_data, non_sw_mask_positions, sw_mask, original_data
    
    def __len__(self):
        return self.shot_num
    
    def __getitem__(self, idx):
        input_data, label_data, non_sw_mask_positions, sw_mask, original_data = self.generate_mask(idx)
        
        return {
            'input_data': input_data,
            'label_data': label_data,
            'non_sw_mask_positions': non_sw_mask_positions,
            'surface_wave_mask': sw_mask,
            'original_data': original_data,
            'shot_idx': idx,
            'is_flipped': self.is_flipped_sample(idx),
            'original_shot_idx': idx if not self.is_flipped_sample(idx) else idx - self.original_shot_num
        }
    
    def visualize_sample(self, idx, figsize=(20, 10)):
        """简化版可视化"""
        sample = self.__getitem__(idx)
        
        input_data = sample['input_data']
        label_data = sample['label_data']
        original_data = sample['original_data']
        non_sw_mask_positions = sample['non_sw_mask_positions']
        sw_mask = sample['surface_wave_mask']
        is_flipped = sample['is_flipped']
        original_shot_idx = sample['original_shot_idx']
        
        current_shot_center = self.get_shot_center_for_idx(idx)
        direct_wave_mask = self.get_direct_wave_mask(current_shot_center)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        vmax = np.percentile(np.abs(original_data), 99)
        
        # 1. 原始数据
        im0 = axes[0].imshow(original_data, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear')
        title_prefix = f'翻转样本 - 原始第{original_shot_idx}炮' if is_flipped else f'原始样本 - 第{idx}炮'
        axes[0].set_title(f'1. 原始数据 - {title_prefix}', fontsize=11, fontweight='bold')
        axes[0].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[0].set_xlabel('道号'); axes[0].set_ylabel('时间采样点')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 绘制速度线
        if self.velocity_regions:
            colors = ['yellow', 'cyan', 'magenta', 'orange', 'purple']
            for i, region in enumerate(self.velocity_regions):
                color = colors[i % len(colors)]
                v_min, v_max = region['velocity_range']
                for v in [v_min, v_max]:
                    if v != 0:
                        traces = np.arange(self.trace_num)
                        offsets = (traces - current_shot_center) * self.dx
                        times = np.abs(offsets / v) if v != 0 else np.zeros_like(offsets)
                        time_indices = times / self.dt
                        if v > 0:
                            valid = traces >= current_shot_center
                        else:
                            valid = traces <= current_shot_center
                        axes[0].plot(traces[valid], time_indices[valid], color=color, linestyle='-', linewidth=1.5)
        
        # 2. 面波区域标识
        sw_mask_display = sw_mask.copy()
        sw_mask_display[~direct_wave_mask] = 0
        im1 = axes[1].imshow(sw_mask_display, aspect='auto', cmap='RdYlGn_r',
                             vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title(f'2. 面波区域标识', fontsize=11, color='green')
        axes[1].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[1].set_xlabel('道号'); axes[1].set_ylabel('时间采样点')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. 损失函数计算位置
        im2 = axes[2].imshow(non_sw_mask_positions.astype(int), aspect='auto', 
                             cmap='Oranges', vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_title(f'3. 损失函数计算位置', fontsize=11, color='red')
        axes[2].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[2].set_xlabel('道号'); axes[2].set_ylabel('时间采样点')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 4. 网络输入
        im3 = axes[3].imshow(input_data, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[3].set_title(f'4. 网络输入', fontsize=11, fontweight='bold', color='blue')
        axes[3].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[3].set_xlabel('道号'); axes[3].set_ylabel('时间采样点')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        # 5. 网络标签
        im4 = axes[4].imshow(label_data, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[4].set_title(f'5. 网络标签', fontsize=11, fontweight='bold', color='green')
        axes[4].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[4].set_xlabel('道号'); axes[4].set_ylabel('时间采样点')
        plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        
        # 6. 输入 - 标签
        diff_input_label = input_data - label_data
        im5 = axes[5].imshow(diff_input_label, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[5].set_title('6. 输入 - 标签', fontsize=11, fontweight='bold', color='red')
        axes[5].axvline(current_shot_center, color='lime', linestyle='--', linewidth=2)
        axes[5].set_xlabel('道号'); axes[5].set_ylabel('时间采样点')
        plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
        
        # 统计信息
        valid_area = direct_wave_mask.sum()
        sw_mask_valid = sw_mask.copy()
        sw_mask_valid[~direct_wave_mask] = 0
        sw_ratio = sw_mask_valid.sum() / valid_area * 100 if valid_area > 0 else 0
        non_sw_mask_ratio = np.sum(non_sw_mask_positions) / valid_area * 100 if valid_area > 0 else 0
        
        flip_status = " [翻转样本]" if is_flipped else " [原始样本]"
        bandpass_info = f" | 带通: {self.bandpass_low}-{self.bandpass_high}Hz" if self.enable_bandpass else ""
        velocity_info = f" | 速度阈值区域: {len(self.velocity_regions)}个" if self.velocity_regions else ""
        info_text = (f'{flip_status} | 面波: {sw_ratio:.2f}% | 非面波掩码: {non_sw_mask_ratio:.2f}%'
                     f'{bandpass_info}{velocity_info}')
        fig.suptitle(info_text, fontsize=13, y=1.01)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig


if __name__ == "__main__":
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # ========== 基于速度的自定义阈值配置 ==========
    # velocity_range: (最小速度, 最大速度) 单位: m/s
    #   正值: 炮点右侧 (道号 > shot_center)
    #   负值: 炮点左侧 (道号 < shot_center)
    # threshold: 该速度范围的能量阈值
    VELOCITY_REGIONS = [
        {'velocity_range': (-1, -50), 'threshold': 0},  # 左侧100-200 m/s
        {'velocity_range': (1, 50), 'threshold': 0},    # 右侧100-200 m/s
    ]
    
    dataset = SeismicMaskDataset_FIELD(
        segy_path=r"D:\桌面\面波压制\训练数据\modified_SN03.sgy",
        shot_center_mode='cyclic',
        shot_center_fixed=24,
        shot_center_start=12,
        shot_center_end=35,
        dt=0.00025,
        non_sw_density_range=(0, 0.03),
        shotnum=168,
        max_time_samples=1200,
        window_width=0.01,
        time_shift=0.002,
        sw_threshold=0.05,  # 默认阈值（未指定速度区域时使用）
        remove_direct=True,
        direct_velocity=1200,
        mute_time0=0.010,
        dx=2,
        taper=20,
        enable_augmentation=False,
        mask_k=0.4,
        # 带通滤波参数
        enable_bandpass=True,
        bandpass_low=20,
        bandpass_high=50,
        sample_rate=4000,
        bandpass_order=4,
        # 基于速度的阈值参数
        velocity_regions=VELOCITY_REGIONS,
        apply_velocity_constraint=False,
        sw_min_velocity=50,
        sw_max_velocity=170
    )                 
    
    print(f"\n数据集总样本数: {len(dataset)}")
    
    sample_index_to_visualize = 12 
    print(f"\n正在生成样本 {sample_index_to_visualize} 的可视化...")
    fig1 = dataset.visualize_sample(sample_index_to_visualize)
    plt.show()