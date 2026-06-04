import numpy as np
import matplotlib.pyplot as plt
import segyio
from torch.utils.data import Dataset
from matplotlib import rcParams

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
        t_duration = t[-1] - t[0]
        n_windows = int(t_duration / self.time_shift)
        time_centers = np.arange(n_windows + 1) * self.time_shift + t[0]
        
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
    
    def identify_surface_wave_multi_region(self, shot_gather, dt, 
                                           custom_regions=None,
                                           default_threshold=0.08):
        """
        多区域自定义阈值面波识别
        
        参数:
            shot_gather: (time_samples, trace_num) 地震数据
            dt: 时间采样间隔
            custom_regions: 自定义区域列表，格式:
                [
                    {
                        'trace_range': (start_trace, end_trace),
                        'time_range': (start_time, end_time),
                        'threshold': 0.03
                    },
                    ...
                ]
            default_threshold: 未指定区域的默认阈值
        """
        time_samples, trace_num = shot_gather.shape
        sw_mask = np.zeros((time_samples, trace_num), dtype=bool)
        
        if custom_regions is None or len(custom_regions) == 0:
            # 无自定义区域，使用默认阈值
            for tr in range(trace_num):
                trace = shot_gather[:, tr]
                mask = self.create_trace_mask(trace, dt, default_threshold)
                sw_mask[:, tr] = mask.astype(bool)
        else:
            # 创建区域阈值映射表
            threshold_map = np.full((time_samples, trace_num), default_threshold)
            
            for region in custom_regions:
                tr_start, tr_end = region['trace_range']
                t_start, t_end = region['time_range']
                threshold = region['threshold']
                
                # 转换为索引
                tr_start = max(0, tr_start)
                tr_end = min(trace_num, tr_end)
                t_start_idx = int(t_start / dt)
                t_end_idx = int(t_end / dt)
                t_start_idx = max(0, t_start_idx)
                t_end_idx = min(time_samples, t_end_idx)
                
                # 设置该区域的阈值
                threshold_map[t_start_idx:t_end_idx, tr_start:tr_end] = threshold
            
            # 对每道应用相应的阈值
            for tr in range(trace_num):
                trace = shot_gather[:, tr]
                time_centers, energies = self.compute_segment_energy(trace, dt)
                
                mask = np.zeros(time_samples, dtype=bool)
                
                for i, center in enumerate(time_centers):
                    t_idx = int(center / dt)
                    if t_idx >= time_samples:
                        continue
                    
                    # 获取该时间点该道的阈值
                    threshold = threshold_map[t_idx, tr]
                    
                    if energies[i] > threshold:
                        t_start = center - self.window_width / 2
                        t_end = center + self.window_width / 2
                        idx_start = max(0, int(t_start / dt))
                        idx_end = min(time_samples, int(t_end / dt))
                        mask[idx_start:idx_end] = True
                
                sw_mask[:, tr] = mask
        
        return sw_mask


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
    地震数据掩码数据集生成器（支持多区域自定义阈值）
    """
    
    def __init__(self, segy_path, shot_center=24, dt=0.00025,
                 non_sw_density_range=(5, 15), 
                 shotnum=700, max_time_samples=1200,
                 window_width=0.01, time_shift=0.002, 
                 sw_threshold=0.08,
                 custom_regions=None,  # 新增参数
                 remove_direct=True, direct_velocity=800, 
                 mute_time0=0.010, dx=2, taper=20,
                 enable_augmentation=True,
                 mask_k=1.0):
        
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
        self.custom_regions = custom_regions  # 保存自定义区域配置
        
        self.analyzer = SeismicEnergyAnalyzer(
            window_width=window_width, 
            time_shift=time_shift
        )
        self.sw_threshold = sw_threshold
        
        print("=" * 70)
        print("正在初始化地震数据掩码数据集（多区域阈值策略）...")
        print("=" * 70)
        self.seismic_data = self.read_segy(segy_path, shotnum)
        
        self.seismic_data = self.seismic_data[:, :max_time_samples, :]
        
        self.original_shot_num, self.time_samples, self.trace_num = self.seismic_data.shape
        
        print(f"\n数据集信息:")
        print(f"  原始炮数: {self.original_shot_num}")
        print(f"  时间采样点: {self.time_samples}")
        print(f"  道数: {self.trace_num}")
        print(f"  炮点位置: 第{self.shot_center}道")
        print(f"\n多区域阈值策略:")
        if custom_regions:
            print(f"  自定义区域数量: {len(custom_regions)}")
            for idx, region in enumerate(custom_regions):
                print(f"    区域{idx+1}: 道{region['trace_range']}, "
                      f"时间{region['time_range']}s, 阈值={region['threshold']}")
        print(f"  默认阈值: {sw_threshold}")
        print(f"\n统一高斯掩码策略:")
        print(f"  【输入】:")
        print(f"    - 面波区域: 全部替换为高斯分布掩码 ([-{self.mask_k}, {self.mask_k}])")
        print(f"    - 非面波区域: 原始数据 + 部分位置替换为高斯分布掩码")
        print(f"  【标签】:")
        print(f"    - 面波区域: 与输入相同的高斯掩码")
        print(f"    - 非面波区域: 原始数据（无掩码）")
        
        if self.enable_augmentation:
            print(f"\n数据增强: 启用 (左右翻转)")
            print(f"  增强后总样本数: {self.original_shot_num * 2}")
        else:
            print(f"\n数据增强: 未启用")
        
        print(f"\n开始预处理数据并识别面波区域...")
        self.processed_data = []
        self.surface_wave_masks = []
        
        for shot_idx in range(self.original_shot_num):
            shot_gather = normalize_traces_per_trace(self.seismic_data[shot_idx])
            
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
            
            # 使用多区域阈值进行面波识别
            sw_mask = self.analyzer.identify_surface_wave_multi_region(
                shot_gather,
                dt=self.dt,
                custom_regions=self.custom_regions,
                default_threshold=self.sw_threshold
            )
            
            self.processed_data.append(shot_gather)
            self.surface_wave_masks.append(sw_mask)
            
            if (shot_idx + 1) % 100 == 0:
                print(f"  已处理原始数据 {shot_idx + 1}/{self.original_shot_num} 炮")
        
        if self.enable_augmentation:
            print(f"\n开始生成翻转数据...")
            for shot_idx in range(self.original_shot_num):
                flipped_gather = np.fliplr(self.processed_data[shot_idx])
                flipped_sw_mask = np.fliplr(self.surface_wave_masks[shot_idx])
                
                self.processed_data.append(flipped_gather)
                self.surface_wave_masks.append(flipped_sw_mask)
                
                if (shot_idx + 1) % 100 == 0:
                    print(f"  已生成翻转数据 {shot_idx + 1}/{self.original_shot_num} 炮")
        
        self.processed_data = np.array(self.processed_data)
        self.surface_wave_masks = np.array(self.surface_wave_masks)
        
        self.shot_num = len(self.processed_data)
        
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
            data = np.zeros((shot_num, time, len_shot), dtype=np.float32)
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
        
        if original_data.dtype not in [np.float32, np.float64]:
             original_data = original_data.astype(np.float32)

        current_shot_center = self.get_shot_center_for_idx(shot_idx)
        
        # 获取直达波以下的有效区域
        direct_wave_mask = self.get_direct_wave_mask(
            current_shot_center, self.direct_velocity, self.mute_time0, self.dx, self.dt
        )
        
        valid_sw_mask = (sw_mask & direct_wave_mask).astype(bool)
        valid_non_sw_mask = (~sw_mask.astype(bool)) & direct_wave_mask
        
        # 生成统一的高斯分布掩码
        gaussian_mask = np.random.normal(0, self.mask_k / 3.0, size=(self.time_samples, self.trace_num)).astype(original_data.dtype)
        gaussian_mask = np.clip(gaussian_mask, -self.mask_k, self.mask_k)
        
        # 生成非面波区域的掩码位置
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
        
        # 构建输入数据
        input_data = original_data.copy()
        input_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        input_data[non_sw_mask_positions] = gaussian_mask[non_sw_mask_positions]
        
        # 构建标签数据
        label_data = original_data.copy()
        label_data[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        
        return input_data, label_data, non_sw_mask_positions, valid_sw_mask, original_data
    
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


def visualize_five_plots_clean(dataset, idx, save_prefix=None, dpi=300):
    """绘制六张独立的图，显示多区域阈值效果"""
    sample = dataset.__getitem__(idx)
    
    original_data = sample['original_data']
    input_data = sample['input_data']
    label_data = sample['label_data']
    sw_mask = sample['surface_wave_mask']
    non_sw_mask_positions = sample['non_sw_mask_positions']
    
    current_shot_center = dataset.get_shot_center_for_idx(idx)
    
    direct_wave_mask = dataset.get_direct_wave_mask(
        current_shot_center, 
        dataset.direct_velocity, 
        dataset.mute_time0, 
        dataset.dx, 
        dataset.dt
    )
    
    sw_mask_valid = sw_mask & direct_wave_mask
    
    vmax = np.percentile(np.abs(original_data), 99)
    
    fig_list = []
    
    # 图1: 原始地震记录
    fig1 = plt.figure(figsize=(10, 18))
    ax1 = fig1.add_subplot(111)
    ax1.imshow(original_data, aspect='auto', cmap='seismic', 
               vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax1.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_list.append(fig1)
    if save_prefix:
        fig1.savefig(f'{save_prefix}_1_original.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # 图2: 面波区域标识（带自定义区域边界）
    fig2 = plt.figure(figsize=(10, 18))
    ax2 = fig2.add_subplot(111)
    sw_display = sw_mask_valid.astype(float)
    ax2.imshow(sw_display, aspect='auto', cmap='bwr', 
               vmin=0, vmax=1, interpolation='nearest')
    
    # # 绘制自定义区域边界
    # if dataset.custom_regions:
    #     colors = ['yellow', 'cyan', 'magenta', 'orange', 'purple']
    #     for idx_region, region in enumerate(dataset.custom_regions):
    #         color = colors[idx_region % len(colors)]
    #         tr_start, tr_end = region['trace_range']
    #         t_start, t_end = region['time_range']
    #         t_start_idx = int(t_start / dataset.dt)
    #         t_end_idx = int(t_end / dataset.dt)
            
    #         from matplotlib.patches import Rectangle
    #         rect = Rectangle((tr_start, t_start_idx), tr_end - tr_start, 
    #                        t_end_idx - t_start_idx, linewidth=3, 
    #                        edgecolor=color, facecolor='none', linestyle='--')
    #         ax2.add_patch(rect)
    
    ax2.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_list.append(fig2)
    if save_prefix:
        fig2.savefig(f'{save_prefix}_2_surface_wave.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # 图3: 面波(红) + 盲扇(绿) 区域标识
    fig3 = plt.figure(figsize=(3, 4))
    ax3 = fig3.add_subplot(111)
    mask_rgb = np.zeros((dataset.time_samples, dataset.trace_num, 3))
    mask_rgb[:, :, 2] = 1.0
    mask_rgb[sw_mask_valid] = [1.0, 0.0, 0.0]
    mask_rgb[non_sw_mask_positions] = [0.0, 1.0, 0.0]
    
    ax3.imshow(mask_rgb, aspect='auto', interpolation='nearest')
    ax3.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_list.append(fig3)
    if save_prefix:
        fig3.savefig(f'{save_prefix}_3_masks_combined.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # 图4: 掩码值图
    fig4 = plt.figure(figsize=(10, 18))
    ax4 = fig4.add_subplot(111)
    
    mask_values = np.zeros_like(original_data)
    mask_values[sw_mask_valid] = input_data[sw_mask_valid]
    mask_values[non_sw_mask_positions] = input_data[non_sw_mask_positions]
    
    mask_vmax = np.percentile(np.abs(mask_values[mask_values != 0]), 99) if np.any(mask_values != 0) else 1.0
    
    ax4.imshow(mask_values, aspect='auto', cmap='seismic', 
               vmin=-mask_vmax, vmax=mask_vmax, interpolation='bilinear')
    ax4.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_list.append(fig4)
    if save_prefix:
        fig4.savefig(f'{save_prefix}_4_mask_values.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # 图5: 网络输入
    fig5 = plt.figure(figsize=(10, 18))
    ax5 = fig5.add_subplot(111)
    ax5.imshow(input_data, aspect='auto', cmap='seismic', 
               vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax5.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_list.append(fig5)
    if save_prefix:
        fig5.savefig(f'{save_prefix}_5_masked_gather_input.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

    # 图6: 训练标签
    fig6 = plt.figure(figsize=(10, 18))
    ax6 = fig6.add_subplot(111)
    ax6.imshow(label_data, aspect='auto', cmap='seismic', 
               vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax6.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    fig_list.append(fig6)
    if save_prefix:
        fig6.savefig(f'{save_prefix}_6_label_gather_sw_masked.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    return fig_list


# ========== 使用示例 ==========
if __name__ == "__main__":
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # 🌟 定义多区域自定义阈值
    CUSTOM_REGIONS = [
        {
            'trace_range': (17, 32),
            'time_range': (0.0, 0.043),
            'threshold': 0.015
        },
        {
            'trace_range': (21, 23),
            'time_range': (0, 0.03),
            'threshold': 0
        },
        {
            'trace_range': (26, 29),
            'time_range': (0, 0.03),
            'threshold': 0
        },
    ]
    
    sample_idx = 0
    
    dataset = SeismicMaskDataset(
        segy_path=r"D:\桌面\面波压制\Synthetic data test2\noised.sgy",
        shot_center=24,
        dt=0.00025,
        non_sw_density_range=(0, 0.05),
        shotnum=1,
        max_time_samples=1200,
        window_width=0.015,
        time_shift=0.001,
        sw_threshold=0.08,
        custom_regions=CUSTOM_REGIONS,  # 传入自定义区域配置
        remove_direct=True,
        direct_velocity=1200,
        mute_time0=0.027,
        dx=2,
        taper=0,
        enable_augmentation=False,
        mask_k=0.5
    )
    
    # 生成六张图
    fig_list = visualize_five_plots_clean(
        dataset=dataset,
        idx=sample_idx,
        save_prefix=r'D:\桌面\面波压制\流程',
        dpi=600
    )
    
    # 显示所有图像
    plt.show()
    
    print(f"\n已成功生成6张图像 (样本索引: {sample_idx})")
    print("---------------------------------------------------------")
    print("图1: 原始地震记录")
    print("图2: 面波区域标识 (红色面波 + 黄色/青色边界显示自定义区域)")
    print("图3: 面波(红) + 盲扇(绿) 标识")
    print("图4: 掩码值图 (高斯噪声值)")
    print("图5: 网络输入 (面波+盲扇掩码)")
    print("图6: 训练标签 (仅面波掩码)")