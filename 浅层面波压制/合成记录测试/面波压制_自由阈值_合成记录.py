import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
from U_Net_CBAM import UNet
# from U_Net import UNet

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


class SeismicDenoiser:
    """地震数据去噪推理器（多区域自定义阈值）"""
    
    def __init__(self, model_path, device=None, 
                 window_width=0.01, time_shift=0.002, sw_threshold=0.08,
                 fixed_mask_value=0.5):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = UNet(in_channels=1, num_classes=1, base_c=64)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.analyzer = SeismicEnergyAnalyzer(
            window_width=window_width,
            time_shift=time_shift
        )
        self.sw_threshold = sw_threshold
        self.fixed_mask_value = fixed_mask_value
    
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
            data = np.zeros((shot_num, time, len_shot))
            for j in range(shot_num):
                data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
            return data
    
    def normalize_traces_per_trace(self, shot_gather):
        """逐道归一化"""
        normalized_gather = np.copy(shot_gather)
        n_samples, n_traces = normalized_gather.shape
        
        for i in range(n_traces):
            trace = normalized_gather[:, i]
            max_amp = np.max(np.abs(trace))
            if max_amp > 0:
                normalized_gather[:, i] = trace / max_amp
        
        return normalized_gather
    
    def remove_direct_wave(self, data, center_trace=64.5, velocity=800, dt=0.00025, 
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
    
    def get_direct_wave_mask(self, time_samples, trace_num, center_trace, 
                             velocity, mute_time0, dx, dt):
        """计算直达波区域掩码（True表示直达波以下的有效区域）"""
        direct_mask = np.zeros((time_samples, trace_num), dtype=bool)
        
        for tr in range(trace_num):
            offset = abs(tr - center_trace)*dx
            t0 = mute_time0 + offset / velocity
            t0_idx = int(t0/dt)
            
            if t0_idx < time_samples:
                direct_mask[t0_idx:, tr] = True
        
        return direct_mask
    
    def apply_velocity_constraint(self, sw_mask, shot_center, dt, dx, 
                                  min_velocity=200, max_velocity=800):
        """应用速度约束"""
        time_samples, trace_num = sw_mask.shape
        velocity_filtered_mask = np.zeros_like(sw_mask, dtype=bool)
        
        for tr in range(trace_num):
            offset = abs(tr - shot_center) * dx
            
            if offset == 0:
                velocity_filtered_mask[:, tr] = sw_mask[:, tr]
                continue
            
            t_min = offset / max_velocity
            t_max = offset / min_velocity
            
            idx_min = int(t_min / dt)
            idx_max = int(t_max / dt)
            
            idx_min = max(0, idx_min)
            idx_max = min(time_samples - 1, idx_max)
            
            if idx_min < time_samples:
                velocity_window = np.zeros(time_samples, dtype=bool)
                velocity_window[idx_min:idx_max+1] = True
                velocity_filtered_mask[:, tr] = sw_mask[:, tr] & velocity_window
        
        return velocity_filtered_mask
    
    def identify_surface_wave_multi_region(self, shot_gather, dt, shot_center,
                                           custom_regions=None,
                                           default_threshold=0.08,
                                           direct_velocity=800, mute_time0=0.010, 
                                           dx=2, apply_velocity_filter=False,
                                           sw_min_velocity=200, sw_max_velocity=800):
        """
        多区域自定义阈值面波识别
        
        参数:
            custom_regions: 自定义区域列表，格式:
                [
                    {
                        'trace_range': (start_trace, end_trace),  # 道号范围
                        'time_range': (start_time, end_time),     # 时间范围(秒)
                        'threshold': 0.03                         # 该区域的阈值
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
                mask = self.analyzer.create_trace_mask(trace, dt, default_threshold)
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
                time_centers, energies = self.analyzer.compute_segment_energy(trace, dt)
                
                mask = np.zeros(time_samples, dtype=bool)
                
                for i, center in enumerate(time_centers):
                    t_idx = int(center / dt)
                    if t_idx >= time_samples:
                        continue
                    
                    # 获取该时间点该道的阈值
                    threshold = threshold_map[t_idx, tr]
                    
                    if energies[i] > threshold:
                        t_start = center - self.analyzer.window_width / 2
                        t_end = center + self.analyzer.window_width / 2
                        idx_start = max(0, int(t_start / dt))
                        idx_end = min(time_samples, int(t_end / dt))
                        mask[idx_start:idx_end] = True
                
                sw_mask[:, tr] = mask
        
        # 应用速度约束
        if apply_velocity_filter:
            sw_mask = self.apply_velocity_constraint(
                sw_mask, shot_center, dt, dx,
                min_velocity=sw_min_velocity,
                max_velocity=sw_max_velocity
            )
        
        # 只保留直达波以下的面波
        direct_wave_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        
        valid_sw_mask = sw_mask & direct_wave_mask
        
        return valid_sw_mask
    
    def replace_surface_wave_with_mask(self, shot_gather, sw_mask, shot_center, 
                                       direct_velocity=800, mute_time0=0.010, 
                                       dx=2, dt=0.00025):
        """将面波区域替换为高斯分布掩码"""
        time_samples, trace_num = shot_gather.shape
        
        direct_wave_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        
        valid_sw_mask = sw_mask & direct_wave_mask
        
        gaussian_mask = np.random.normal(0, self.fixed_mask_value / 3.0, 
                                        size=(time_samples, trace_num))
        gaussian_mask = np.clip(gaussian_mask, -self.fixed_mask_value, 
                               self.fixed_mask_value)
        
        masked_shot = shot_gather.copy()
        masked_shot[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        
        return masked_shot
    
    def boost_surface_wave_energy(self, denoised_data, sw_mask, 
                                  shot_center, energy_boost_factor=1.5,
                                  direct_velocity=800, 
                                  mute_time0=0.010, dx=2, dt=0.00025):
        """增强面波区域的能量"""
        time_samples, trace_num = denoised_data.shape
        
        direct_wave_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        
        valid_sw_mask = sw_mask & direct_wave_mask
        
        boosted_data = denoised_data.copy()
        
        if not valid_sw_mask.any():
            return boosted_data
        
        boosted_data[valid_sw_mask] = denoised_data[valid_sw_mask] * energy_boost_factor
        
        return boosted_data
    
    def preprocess_shot(self, shot_gather, shot_center=22, remove_direct=True,
                        direct_velocity=800, dt=0.00025, mute_time0=0.010, 
                        dx=2, taper=20, norm = True):
        """预处理单炮数据"""
        if norm:
            processed = self.normalize_traces_per_trace(shot_gather)
        else:
            processed = shot_gather

        if remove_direct:
            processed = self.remove_direct_wave(
                processed,
                center_trace=shot_center,
                velocity=direct_velocity,
                dt=dt,
                mute_time0=mute_time0,
                dx=dx,
                taper=taper
            )
        
        return processed
    
    @torch.no_grad()
    def denoise_shot(self, shot_gather, sw_mask=None, shot_center=22, 
                     replace_non_sw_with_input=True,
                     boost_sw_energy=True,
                     energy_boost_factor=1.5,
                     remove_direct_from_output=True,
                     direct_velocity=800, mute_time0=0.010, dx=2, dt=0.00025,
                     taper=20):
        """
        对单炮数据进行去噪
        
        参数:
            shot_gather: 输入的炮集数据（已预处理）
            sw_mask: 面波掩码
            shot_center: 炮点位置
            replace_non_sw_with_input: 是否将非面波区域替换为输入数据
            boost_sw_energy: 是否增强面波能量
            energy_boost_factor: 能量增强因子
            remove_direct_from_output: 是否从输出中去除直达波
            direct_velocity: 直达波速度
            mute_time0: 静校正时间
            dx: 道间距
            dt: 采样间隔
            taper: 锥形渐变长度
        """
        input_tensor = torch.from_numpy(shot_gather).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        output = self.model(input_tensor)
        denoised = output.squeeze().cpu().numpy()
        
        time_samples, trace_num = denoised.shape
        
        # 获取直达波掩码
        direct_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        
        # 增强面波能量
        if boost_sw_energy and sw_mask is not None:
            denoised = self.boost_surface_wave_energy( 
                denoised, sw_mask, shot_center, energy_boost_factor, 
                direct_velocity, mute_time0, dx, dt
            )
        
        # 处理非面波区域
        if replace_non_sw_with_input and sw_mask is not None:
            # 替换非面波区域为输入数据
            valid_sw_mask = sw_mask & direct_mask
            non_sw_mask = direct_mask & (~valid_sw_mask)
            denoised[non_sw_mask] = shot_gather[non_sw_mask]
            # 替换后，直达波以上的部分已经为0（因为shot_gather已预处理过）
        else:
            # 不替换时，需要手动将直达波以上的部分置零
            denoised[~direct_mask] = 0
        
        # 从输出去除直达波（可选）
        if remove_direct_from_output:
            denoised = self.remove_direct_wave(
                denoised,
                center_trace=shot_center,
                velocity=direct_velocity,
                dt=dt,
                mute_time0=mute_time0,
                dx=dx,
                taper=taper
            )
        
        return denoised, None
    
    def save_shot_to_npy(self, data, output_path):
        """保存单炮数据为numpy格式"""
        np.save(output_path, data)
    
    def plot_comparison(self, original, network_input, denoised, sw_mask, 
                        shot_idx, shot_center=22, custom_regions=None,
                        direct_velocity=800, mute_time0=0.010, dx=2, dt=0.00025,
                        save_path=None, figsize=(18, 12)):
        
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        time_samples, trace_num = original.shape
        direct_wave_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        vmax = np.percentile(np.abs(original), 99)
        
        # 子图1: 预处理后的原始数据
        im0 = axes[0, 0].imshow(original, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0, 0].set_title(f'预处理后原始数据 - 第{shot_idx}炮', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('道号')
        axes[0, 0].set_ylabel('时间采样点')
        axes[0, 0].axvline(shot_center, color='lime', linestyle='--', linewidth=2, label='炮点')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 子图2: 面波区域标识 + 自定义区域边界
        sw_mask_display = sw_mask.copy()
        sw_mask_display[~direct_wave_mask] = 0
        im1 = axes[0, 1].imshow(sw_mask_display, aspect='auto', cmap='RdYlGn_r',
                                vmin=0, vmax=1, interpolation='nearest')
        axes[0, 1].set_title('面波区域标识（多区域阈值）', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('道号')
        axes[0, 1].set_ylabel('时间采样点')
        axes[0, 1].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        
        # 绘制自定义区域边界
        if custom_regions:
            colors = ['yellow', 'cyan', 'magenta', 'orange', 'purple']
            for idx, region in enumerate(custom_regions):
                color = colors[idx % len(colors)]
                tr_start, tr_end = region['trace_range']
                t_start, t_end = region['time_range']
                t_start_idx = int(t_start / dt)
                t_end_idx = int(t_end / dt)
                
                from matplotlib.patches import Rectangle
                rect = Rectangle((tr_start, t_start_idx), tr_end - tr_start, 
                               t_end_idx - t_start_idx, linewidth=2, 
                               edgecolor=color, facecolor='none', linestyle='--')
                axes[0, 1].add_patch(rect)
                axes[0, 1].text(tr_start + 1, t_start_idx + 20, 
                              f'区域{idx+1}\n阈值={region["threshold"]:.3f}',
                              color=color, fontsize=8, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 子图3: 网络输入
        im2 = axes[0, 2].imshow(network_input, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0, 2].set_title('网络输入', fontsize=11, fontweight='bold')
        axes[0, 2].set_xlabel('道号')
        axes[0, 2].set_ylabel('时间采样点')
        axes[0, 2].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 子图4: 网络输入 + 面波区域叠加
        axes[1, 0].imshow(network_input, aspect='auto', cmap='gray',
                          vmin=-vmax, vmax=vmax, interpolation='bilinear')
        masked_sw = np.ma.masked_where(sw_mask_display == 0, sw_mask_display)
        axes[1, 0].imshow(masked_sw, aspect='auto', cmap='Reds', 
                         alpha=0.4, vmin=0, vmax=1, interpolation='nearest')
        axes[1, 0].set_title('网络输入 + 面波区域叠加', fontsize=11)
        axes[1, 0].set_xlabel('道号')
        axes[1, 0].set_ylabel('时间采样点')
        axes[1, 0].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        
        # 子图5: 最终结果
        im5 = axes[1, 1].imshow(denoised, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[1, 1].set_title('最终结果', fontsize=11, fontweight='bold', color='green')
        axes[1, 1].set_xlabel('道号')
        axes[1, 1].set_ylabel('时间采样点')
        axes[1, 1].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im5, ax=axes[1, 1])
        
        # 子图6: 总体差异
        total_diff = original - denoised
        vmax_diff = np.percentile(np.abs(total_diff), 99)
        im7 = axes[1, 2].imshow(total_diff, aspect='auto', cmap='seismic',
                                vmin=-vmax_diff, vmax=vmax_diff, interpolation='bilinear')
        axes[1, 2].set_title('总体差异（原始 - 最终）', fontsize=11)
        axes[1, 2].set_xlabel('道号')
        axes[1, 2].set_ylabel('时间采样点')
        axes[1, 2].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im7, ax=axes[1, 2])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
        
    # ========== 基础配置 ==========
    model_path = r"D:\桌面\面波压制\model\model_syn_1027\model_epoch_500.pth"
    input_segy_path = r"D:\桌面\面波压制\训练数据\dataset1_260.sgy"
    output_dir = r"D:\桌面\面波压制\合成记录结果\DMSSL"
    os.makedirs(output_dir, exist_ok=True)

    # ========== 🌟 多区域自定义阈值配置 🌟 ==========
    # 格式说明:
    # - trace_range: (起始道号, 结束道号)
    # - time_range: (起始时间(秒), 结束时间(秒))
    # - threshold: 该区域的能量阈值
    
    CUSTOM_REGIONS = [
        {
            'trace_range': (17, 32),      # 近炮点区域
            'time_range': (0.0, 0.037),    # 浅层时间
            'threshold': 0.015              # 低阈值
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
    
    DEFAULT_THRESHOLD = 0.5           # 未指定区域的默认阈值
    
    # ========== 🆕 直达波控制参数 🆕 ==========
    REMOVE_DIRECT_FROM_INPUT = True    # 网络输入是否去除直达波
    REMOVE_DIRECT_FROM_OUTPUT = True   # 网络输出是否去除直达波
    
    # ========== 其他参数 ==========
    USE_MASK_FOR_SURFACE_WAVE = True
    BOOST_SURFACE_WAVE_ENERGY = True
    ENERGY_BOOST_FACTOR = 8
    REPLACE_NON_SW_WITH_INPUT = True   # 是否将非面波区域替换为输入
    APPLY_VELOCITY_CONSTRAINT = True
    SW_MIN_VELOCITY = 10
    SW_MAX_VELOCITY = 170
    norm = False
    
    target_shot_idx = 0
    shot_center = 24
    shotnum = 260
    max_time_samples = 1200
    direct_velocity = 1200
    dt = 0.00025
    mute_time0 = 0.027
    dx = 2
    taper = 0
    window_width = 0.015
    time_shift = 0.0005
    fixed_mask_value = 0.5
    
    # ========== 初始化 ==========
    denoiser = SeismicDenoiser(
        model_path,
        window_width=window_width,
        time_shift=time_shift,
        sw_threshold=DEFAULT_THRESHOLD,
        fixed_mask_value=fixed_mask_value
    )
    
    # ========== 读取和预处理 ==========
    seismic_data = denoiser.read_segy(input_segy_path, shotnum=shotnum)
    seismic_data = seismic_data[:, :max_time_samples, :]
    single_shot = seismic_data[target_shot_idx]
    
    # 预处理：归一化 + 可选的直达波去除
    preprocessed_data = denoiser.preprocess_shot(
        single_shot,
        shot_center=shot_center,
        remove_direct=REMOVE_DIRECT_FROM_INPUT,  # 使用配置参数
        direct_velocity=direct_velocity,
        dt=dt,
        mute_time0=mute_time0,
        dx=dx,
        taper=taper,
        norm = norm
    )
    
    # ========== 多区域面波识别 ==========
    sw_mask = denoiser.identify_surface_wave_multi_region(
        preprocessed_data,
        dt=dt,
        shot_center=shot_center,
        custom_regions=CUSTOM_REGIONS,
        default_threshold=DEFAULT_THRESHOLD,
        direct_velocity=direct_velocity,
        mute_time0=mute_time0,
        dx=dx,
        apply_velocity_filter=APPLY_VELOCITY_CONSTRAINT,
        sw_min_velocity=SW_MIN_VELOCITY,
        sw_max_velocity=SW_MAX_VELOCITY
    )
    
    # ========== 准备网络输入 ==========
    if USE_MASK_FOR_SURFACE_WAVE:
        network_input = denoiser.replace_surface_wave_with_mask(
            preprocessed_data, sw_mask, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
    else:
        network_input = preprocessed_data.copy()
    
    # ========== 去噪 ==========
    denoised_shot, _ = denoiser.denoise_shot(
        network_input, sw_mask, shot_center,
        replace_non_sw_with_input=REPLACE_NON_SW_WITH_INPUT,
        boost_sw_energy=BOOST_SURFACE_WAVE_ENERGY,
        energy_boost_factor=ENERGY_BOOST_FACTOR,
        remove_direct_from_output=REMOVE_DIRECT_FROM_OUTPUT,  # 使用配置参数
        direct_velocity=direct_velocity,
        mute_time0=mute_time0,
        dx=dx, dt=dt, taper=taper
    )
    
    noise = preprocessed_data - denoised_shot
    
    # ========== 保存结果 ==========
    denoiser.save_shot_to_npy(denoised_shot, 
        os.path.join(output_dir, f"denoised_shot_{target_shot_idx}.npy"))
    denoiser.save_shot_to_npy(preprocessed_data, 
        os.path.join(output_dir, f"original_shot_{target_shot_idx}.npy"))
    denoiser.save_shot_to_npy(network_input, 
        os.path.join(output_dir, f"input_shot_{target_shot_idx}.npy"))
    denoiser.save_shot_to_npy(sw_mask, 
        os.path.join(output_dir, f"sw_mask_shot_{target_shot_idx}.npy"))
    denoiser.save_shot_to_npy(noise, 
        os.path.join(output_dir, f"noise_{target_shot_idx}.npy"))
    
    # ========== 绘图 ==========
    comparison_path = os.path.join(output_dir, 
        f"comparison_shot_{target_shot_idx}.png")
    
    denoiser.plot_comparison(
        preprocessed_data, network_input, denoised_shot, sw_mask,
        target_shot_idx, shot_center, CUSTOM_REGIONS,
        direct_velocity, mute_time0, dx, dt,
        save_path=comparison_path
    )
    
    print(f"\n处理完成！结果保存在: {output_dir}")
    print(f"\n配置信息:")
    print(f"  - 网络输入去除直达波: {REMOVE_DIRECT_FROM_INPUT}")
    print(f"  - 网络输出去除直达波: {REMOVE_DIRECT_FROM_OUTPUT}")
    print(f"  - 非面波区域替换为输入: {REPLACE_NON_SW_WITH_INPUT}")
    print(f"  - 面波能量增强: {BOOST_SURFACE_WAVE_ENERGY} (因子: {ENERGY_BOOST_FACTOR})")