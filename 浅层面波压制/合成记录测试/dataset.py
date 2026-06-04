import numpy as np
import matplotlib.pyplot as plt
import segyio
from torch.utils.data import Dataset

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


class SeismicMaskDataset(Dataset):
    """
    地震数据掩码数据集生成器（统一高斯掩码策略）
    
    新策略说明：
    1. 输入（Input）:
       - 面波区域: 全部替换为高斯分布掩码 [-k, k]
       - 非面波区域: 原始数据 + 部分位置替换为高斯分布掩码 [-k, k]
    
    2. 标签（Label）:
       - 面波区域: 与输入相同的高斯分布掩码
       - 非面波区域: 原始数据（无掩码）
    
    3. 损失计算:
       - 仅在非面波区域的掩码位置计算损失
    """
    
    def __init__(self, segy_path, shot_center=24, dt=0.00025,
                 non_sw_density_range=(5, 15),  # 非面波区域活跃像素密度
                 shotnum=700, max_time_samples=1200,
                 # 面波识别参数
                 window_width=0.01, time_shift=0.002, sw_threshold=0.08,
                 # 直达波去除参数
                 remove_direct=True, direct_velocity=800, 
                 mute_time0=0.010, dx=2, taper=20,
                 # 数据增强参数
                 enable_augmentation=True,
                 # 掩码参数（统一使用高斯分布）
                 mask_k=1.0):  # 掩码值范围 [-k, k]，面波和非面波统一使用
        
        self.shot_center = shot_center
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
        
        # 创建面波分析器
        self.analyzer = SeismicEnergyAnalyzer(
            window_width=window_width, 
            time_shift=time_shift
        )
        self.sw_threshold = sw_threshold
        
        # 读取SEGY数据
        print("=" * 70)
        print("正在初始化地震数据掩码数据集（统一高斯掩码策略）...")
        print("=" * 70)
        self.seismic_data = self.read_segy(segy_path, shotnum)
        
        # 只保留前max_time_samples个采样点
        self.seismic_data = self.seismic_data[:, :max_time_samples, :]
        
        self.original_shot_num, self.time_samples, self.trace_num = self.seismic_data.shape
        
        print(f"\n数据集信息:")
        print(f"  原始炮数: {self.original_shot_num}")
        print(f"  时间采样点: {self.time_samples}")
        print(f"  道数: {self.trace_num}")
        print(f"  炮点位置: 第{self.shot_center}道")
        print(f"\n统一高斯掩码策略:")
        print(f"  【输入】:")
        print(f"    - 面波区域: 全部替换为高斯分布掩码 ([-{self.mask_k}, {self.mask_k}])")
        print(f"    - 非面波区域: 原始数据 + 部分位置替换为高斯分布掩码 ([-{self.mask_k}, {self.mask_k}])")
        print(f"  【标签】:")
        print(f"    - 面波区域: 与输入相同的高斯掩码")
        print(f"    - 非面波区域: 原始数据（无掩码）")
        print(f"  【损失计算】:")
        print(f"    - 仅在非面波区域的掩码位置计算")
        
        if self.enable_augmentation:
            print(f"\n数据增强: 启用 (左右翻转)")
            print(f"  增强后总样本数: {self.original_shot_num * 2}")
        else:
            print(f"\n数据增强: 未启用")
        
        # 预处理所有炮集并识别面波区域
        print(f"\n开始预处理数据并识别面波区域...")
        self.processed_data = []
        self.surface_wave_masks = []
        
        # 处理原始数据
        for shot_idx in range(self.original_shot_num):
            # 归一化
            shot_gather = normalize_traces_per_trace(self.seismic_data[shot_idx])
            
            # 去除直达波
            if self.remove_direct:
                shot_gather = remove_direct_wave(
                    shot_gather,
                    center_trace=self.shot_center,
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
            
            if (shot_idx + 1) % 100 == 0:
                print(f"  已处理原始数据 {shot_idx + 1}/{self.original_shot_num} 炮")
        
        # 数据增强：添加左右翻转的数据
        if self.enable_augmentation:
            print(f"\n开始生成翻转数据...")
            for shot_idx in range(self.original_shot_num):
                # 左右翻转
                flipped_gather = np.fliplr(self.processed_data[shot_idx])
                flipped_sw_mask = np.fliplr(self.surface_wave_masks[shot_idx])
                
                self.processed_data.append(flipped_gather)
                self.surface_wave_masks.append(flipped_sw_mask)
                
                if (shot_idx + 1) % 100 == 0:
                    print(f"  已生成翻转数据 {shot_idx + 1}/{self.original_shot_num} 炮")
        
        self.processed_data = np.array(self.processed_data)
        self.surface_wave_masks = np.array(self.surface_wave_masks)
        
        # 更新实际的炮数（包含增强后的数据）
        self.shot_num = len(self.processed_data)
        
        # 计算翻转后的炮点位置
        self.flipped_shot_center = self.trace_num - 1 - self.shot_center
        
        print(f"\n数据预处理完成！")
        print(f"  最终数据集大小: {self.shot_num} 炮")
        if self.enable_augmentation:
            print(f"  (原始: {self.original_shot_num} + 翻转: {self.original_shot_num})")
        print("=" * 70)
    
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
            data = np.zeros((shot_num, time, len_shot))
            for j in range(shot_num):
                data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
            return data
    
    def is_flipped_sample(self, idx):
        """判断索引是否对应翻转样本"""
        if not self.enable_augmentation:
            return False
        return idx >= self.original_shot_num
    
    def get_shot_center_for_idx(self, idx):
        """获取对应索引的炮点位置"""
        if self.is_flipped_sample(idx):
            return self.flipped_shot_center
        else:
            return self.shot_center
    
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
        """
        计算直达波区域掩码
        返回: 布尔数组，True表示在直达波以下（可以添加掩码的区域）
        """
        direct_mask = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        
        for tr in range(self.trace_num):
            offset = abs(tr - center_trace)*dx
            t0 = mute_time0 + offset / velocity
            t0_idx = int(t0/dt)
            
            if t0_idx < self.time_samples:
                direct_mask[t0_idx:, tr] = True
        
        return direct_mask
    
    def generate_mask(self, shot_idx):
        # 获取预处理后的数据和面波掩码
        original_data = self.processed_data[shot_idx].copy()
        sw_mask = self.surface_wave_masks[shot_idx]
        
        # 🚨 新增检查点：确认数据类型 (理论上 read_segy 修正后这里应该是 float32)
        if original_data.dtype not in [np.float32, np.float64]:
             print(f"⚠️ 警告：original_data 的 dtype 为 {original_data.dtype}，正在强制转换为 float32。")
             original_data = original_data.astype(np.float32)

        # 根据是否为翻转样本，使用对应的炮点位置
        current_shot_center = self.get_shot_center_for_idx(shot_idx)
        
        # 获取直达波以下的有效区域
        direct_wave_mask = self.get_direct_wave_mask(
            current_shot_center, self.direct_velocity, self.mute_time0, self.dx, self.dt
        )
        
        # valid_sw_mask 和 valid_non_sw_mask 保持不变
        valid_sw_mask = (sw_mask & direct_wave_mask).astype(bool) # 确保是 bool 类型
        valid_non_sw_mask = (~sw_mask.astype(bool)) & direct_wave_mask
        
        # ========== 生成统一的高斯分布掩码 ==========
        gaussian_mask = np.random.normal(0, self.mask_k / 3.0, size=(self.time_samples, self.trace_num)).astype(original_data.dtype) # 确保高斯掩码类型与数据一致
        gaussian_mask = np.clip(gaussian_mask, -self.mask_k, self.mask_k)
        
        # ========== 生成非面波区域的掩码位置 ==========
        non_sw_mask_positions = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        
        # ... (非面波掩码生成逻辑保持不变)
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
        
        # ========== 构建输入数据（Input）==========
        input_data = original_data.copy()
        
        # 1. 面波区域：全部替换为高斯分布掩码
        input_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        
        # 2. 非面波区域：在掩码位置替换为高斯分布掩码（统一！）
        input_data[non_sw_mask_positions] = gaussian_mask[non_sw_mask_positions]
        
        # ========== 构建标签数据（Label）==========
        label_data = original_data.copy()
        
        # 1. 面波区域：使用与输入相同的高斯掩码
        label_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        
        # 2. 非面波区域：保持原始数据不变
        
        return input_data, label_data, non_sw_mask_positions, valid_sw_mask, original_data
    
    def __len__(self):
        return self.shot_num
    
    def __getitem__(self, idx):
        input_data, label_data, non_sw_mask_positions, sw_mask, original_data = self.generate_mask(idx)
        
        return {
            'input_data': input_data,       # 网络输入
            'label_data': label_data,       # 网络标签
            'non_sw_mask_positions': non_sw_mask_positions,  # 损失计算位置
            'surface_wave_mask': sw_mask,
            'original_data': original_data,  # 原始数据（用于可视化）
            'shot_idx': idx,
            'is_flipped': self.is_flipped_sample(idx),
            'original_shot_idx': idx if not self.is_flipped_sample(idx) else idx - self.original_shot_num
        }
    
    def visualize_sample(self, idx, figsize=(18, 12)):
    
        sample = self.__getitem__(idx)
        
        input_data = sample['input_data']
        label_data = sample['label_data']
        original_data = sample['original_data']
        non_sw_mask_positions = sample['non_sw_mask_positions']
        sw_mask = sample['surface_wave_mask']
        is_flipped = sample['is_flipped']
        original_shot_idx = sample['original_shot_idx']
        
        current_shot_center = self.get_shot_center_for_idx(idx)
        direct_wave_mask = self.get_direct_wave_mask(
            current_shot_center, self.direct_velocity, self.mute_time0, self.dx, self.dt
        )
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 计算显示范围
        vmax = np.percentile(np.abs(original_data), 99)
        
        # 炮点信息
        title_prefix = f'翻转样本 - 原始第{original_shot_idx}炮' if is_flipped else f'原始样本 - 第{idx}炮'
        
        # ========== 1. 原始数据 ==========
        im0 = axes[0, 0].imshow(original_data, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0, 0].set_title(f'1. 原始数据\n{title_prefix}', fontsize=10, fontweight='bold')
        axes[0, 0].axvline(current_shot_center, color='lime', linestyle='--', linewidth=1.5, label='炮点')
        axes[0, 0].set_xlabel('道号', fontsize=9)
        axes[0, 0].set_ylabel('时间采样点', fontsize=9)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # ========== 2. 面波区域识别 ==========
        sw_mask_display = sw_mask.copy()
        sw_mask_display[~direct_wave_mask] = 0
        im1 = axes[0, 1].imshow(sw_mask_display, aspect='auto', cmap='RdYlGn_r',
                                vmin=0, vmax=1, interpolation='nearest')
        axes[0, 1].set_title('2. 面波区域识别\n(有效区域内)', fontsize=10, fontweight='bold')
        axes[0, 1].axvline(current_shot_center, color='lime', linestyle='--', linewidth=1.5)
        axes[0, 1].set_xlabel('道号', fontsize=9)
        axes[0, 1].set_ylabel('时间采样点', fontsize=9)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # ========== 3. 非面波区域高斯掩码位置 ==========
        im2 = axes[0, 2].imshow(non_sw_mask_positions.astype(int), aspect='auto', 
                                cmap='Oranges', vmin=0, vmax=1, interpolation='nearest')
        axes[0, 2].set_title('3. 非面波区域\n高斯掩码位置', fontsize=10, fontweight='bold')
        axes[0, 2].axvline(current_shot_center, color='lime', linestyle='--', linewidth=1.5)
        axes[0, 2].set_xlabel('道号', fontsize=9)
        axes[0, 2].set_ylabel('时间采样点', fontsize=9)
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # ========== 4. 网络输入 ==========
        im3 = axes[1, 0].imshow(input_data, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[1, 0].set_title('4. 网络输入 (Input)\n面波+非面波掩码位置高斯化', 
                            fontsize=10, fontweight='bold', color='blue')
        axes[1, 0].axvline(current_shot_center, color='lime', linestyle='--', linewidth=1.5)
        axes[1, 0].set_xlabel('道号', fontsize=9)
        axes[1, 0].set_ylabel('时间采样点', fontsize=9)
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # ========== 5. 网络标签 ==========
        im4 = axes[1, 1].imshow(label_data, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[1, 1].set_title('5. 网络标签 (Label)\n面波高斯化+非面波原始数据', 
                            fontsize=10, fontweight='bold', color='green')
        axes[1, 1].axvline(current_shot_center, color='lime', linestyle='--', linewidth=1.5)
        axes[1, 1].set_xlabel('道号', fontsize=9)
        axes[1, 1].set_ylabel('时间采样点', fontsize=9)
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # ========== 6. 损失函数计算位置 ==========
        # 使用与非面波掩码相同的显示方式
        im5 = axes[1, 2].imshow(non_sw_mask_positions.astype(int), aspect='auto', 
                                cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        axes[1, 2].set_title('6. 损失函数计算位置\n(非面波掩码位置)', 
                            fontsize=10, fontweight='bold', color='red')
        axes[1, 2].axvline(current_shot_center, color='lime', linestyle='--', linewidth=1.5)
        axes[1, 2].set_xlabel('道号', fontsize=9)
        axes[1, 2].set_ylabel('时间采样点', fontsize=9)
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # 统计信息
        valid_area = direct_wave_mask.sum()
        sw_ratio = sw_mask_display.sum() / valid_area * 100 if valid_area > 0 else 0
        non_sw_mask_ratio = np.sum(non_sw_mask_positions) / valid_area * 100 if valid_area > 0 else 0
        
        flip_status = "[翻转样本]" if is_flipped else "[原始样本]"
        info_text = (f'{flip_status} | 面波区域占比: {sw_ratio:.2f}% | '
                    f'非面波掩码占比: {non_sw_mask_ratio:.2f}% | '
                    f'高斯掩码范围: [-{self.mask_k}, {self.mask_k}]')
        fig.suptitle(info_text, fontsize=12, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

# ========== 使用示例 ==========
if __name__ == "__main__":
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # 创建数据集（统一高斯掩码策略）
    dataset = SeismicMaskDataset(
        segy_path=r"D:\桌面\面波压制\data\dataset1_260.sgy",
        shot_center=24,
        dt=0.00025,
        non_sw_density_range=(0, 0.05),  # 非面波区域活跃像素密度
        shotnum=260,
        max_time_samples=1200,
        # 面波识别参数
        window_width=0.01,
        time_shift=0.002,
        sw_threshold=0.08,
        # 直达波去除参数
        remove_direct=True,
        direct_velocity=1200,
        mute_time0=0.010,
        dx=2,
        taper=20,
        # 数据增强参数
        enable_augmentation=False,
        # 掩码参数（统一高斯分布）
        mask_k=0.4  # 面波和非面波统一使用 [-0.4, 0.4] 高斯分布
    )
    
    print(f"\n数据集总样本数: {len(dataset)}")
    print(f"原始样本: 0-{dataset.original_shot_num-1}")
    
    # 可视化原始样本
    print("\n正在生成原始样本可视化...")
    fig1 = dataset.visualize_sample(13)
    plt.show()