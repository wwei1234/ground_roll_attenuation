import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
from U_Net_CBAM import UNet
# from U_Net import UNet

class SeismicEnergyAnalyzer:
    """地震数据能量片段分析器（与训练时相同）"""
    
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
class SeismicDenoiser:
    """地震数据去噪推理器（带面波能量增强和速度约束）"""
    
    def __init__(self, model_path, device=None, 
                 window_width=0.01, time_shift=0.002, sw_threshold=0.08,
                 fixed_mask_value=0.5):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = UNet(in_channels=1, num_classes=1, base_c=64)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载: {model_path}")
        
        # 面波识别器
        self.analyzer = SeismicEnergyAnalyzer(
            window_width=window_width,
            time_shift=time_shift
        )
        self.sw_threshold = sw_threshold
        self.fixed_mask_value = fixed_mask_value
        
        print(f"面波识别参数: 窗宽={window_width}, 时移={time_shift}, 阈值={sw_threshold}")
        print(f"固定掩码值: {fixed_mask_value}")
    
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
        """
        应用速度约束，过滤掉不符合速度范围的面波区域
        
        参数:
            sw_mask: 初步识别的面波掩码 (time_samples, trace_num)
            shot_center: 炮点位置（道号）
            dt: 时间采样间隔
            dx: 道间距
            min_velocity: 最小速度 (m/s)
            max_velocity: 最大速度 (m/s)
        
        返回:
            velocity_filtered_mask: 速度约束后的面波掩码
        """
        time_samples, trace_num = sw_mask.shape
        velocity_filtered_mask = np.zeros_like(sw_mask, dtype=bool)
        
        print(f"\n应用速度约束: {min_velocity} m/s <= v <= {max_velocity} m/s")
        
        # 对每个道进行处理
        for tr in range(trace_num):
            # 计算偏移距
            offset = abs(tr - shot_center) * dx
            
            if offset == 0:
                # 炮点位置，直接保留原始掩码
                velocity_filtered_mask[:, tr] = sw_mask[:, tr]
                continue
            
            # 计算该道在速度范围内的时间窗口
            t_min = offset / max_velocity  # 最大速度对应的最早到达时间
            t_max = offset / min_velocity  # 最小速度对应的最晚到达时间
            
            # 转换为采样点索引
            idx_min = int(t_min / dt)
            idx_max = int(t_max / dt)
            
            # 确保索引在有效范围内
            idx_min = max(0, idx_min)
            idx_max = min(time_samples - 1, idx_max)
            
            # 在时间窗口内保留原始面波掩码
            if idx_min < time_samples:
                # 创建速度窗口掩码
                velocity_window = np.zeros(time_samples, dtype=bool)
                velocity_window[idx_min:idx_max+1] = True
                
                # 只保留同时满足能量阈值和速度约束的点
                velocity_filtered_mask[:, tr] = sw_mask[:, tr] & velocity_window
        
        # 统计过滤效果
        original_pixels = sw_mask.sum()
        filtered_pixels = velocity_filtered_mask.sum()
        removed_pixels = original_pixels - filtered_pixels
        removal_rate = (removed_pixels / original_pixels * 100) if original_pixels > 0 else 0
        
        print(f"  原始面波像素: {original_pixels}")
        print(f"  速度约束后像素: {filtered_pixels}")
        print(f"  移除像素: {removed_pixels} ({removal_rate:.1f}%)")
        
        return velocity_filtered_mask
    
    def identify_surface_wave(self, shot_gather, dt, shot_center, 
                              direct_velocity=800, mute_time0=0.010, dx=2,
                              apply_velocity_filter=True,
                              sw_min_velocity=200, sw_max_velocity=800):
        """
        识别面波区域（带速度约束）
        
        参数:
            apply_velocity_filter: 是否应用速度约束过滤
            sw_min_velocity: 面波最小速度
            sw_max_velocity: 面波最大速度
        """
        time_samples, trace_num = shot_gather.shape
        
        # 第一步：基于能量阈值的初步识别
        sw_mask = self.analyzer.batch_identify_surface_wave(
            shot_gather, 
            dt, 
            self.sw_threshold
        )
        
        print(f"  初步能量识别完成: {sw_mask.sum()} 个像素")
        
        # 第二步：应用速度约束（可选）
        if apply_velocity_filter:
            sw_mask = self.apply_velocity_constraint(
                sw_mask.astype(bool),
                shot_center,
                dt,
                dx,
                min_velocity=sw_min_velocity,
                max_velocity=sw_max_velocity
            )
        
        # 第三步：只保留直达波以下的面波
        direct_wave_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        
        valid_sw_mask = sw_mask.astype(bool) & direct_wave_mask
        
        print(f"  最终有效面波区域: {valid_sw_mask.sum()} 个像素")
        
        return valid_sw_mask.astype(bool)
    
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
        
        gaussian_mask = np.random.normal(0, self.fixed_mask_value / 3.0, size=(time_samples, trace_num))
        gaussian_mask = np.clip(gaussian_mask, -self.fixed_mask_value, self.fixed_mask_value)
        
        masked_shot = shot_gather.copy()
        masked_shot[valid_sw_mask] = gaussian_mask[valid_sw_mask]
        
        print(f"  面波区域替换完成: {valid_sw_mask.sum()} 个像素被替换为高斯掩码")
        
        return masked_shot
    
    def boost_surface_wave_energy(self, denoised_data, sw_mask, 
                                  shot_center, energy_boost_factor=1.5,
                                  direct_velocity=800, 
                                  mute_time0=0.010, dx=2, dt=0.00025):
        """增强面波区域的能量（手动指定增强系数）"""
        time_samples, trace_num = denoised_data.shape
        
        direct_wave_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        
        valid_sw_mask = sw_mask & direct_wave_mask
        
        boosted_data = denoised_data.copy()
        
        if not valid_sw_mask.any():
            print("  未发现有效面波区域，跳过能量增强")
            return boosted_data
        
        print(f"\n开始面波能量增强（手动系数模式）...")
        print(f"  增强系数: {energy_boost_factor:.2f}x")
        
        boosted_data[valid_sw_mask] = denoised_data[valid_sw_mask] * energy_boost_factor
        
        enhanced_pixels = valid_sw_mask.sum()
        
        print(f"  能量增强完成:")
        print(f"    - 处理像素数: {enhanced_pixels}")
        print(f"    - 应用增强系数: {energy_boost_factor:.2f}x")
        
        return boosted_data
    
    def preprocess_shot(self, shot_gather, shot_center=22, remove_direct=True,
                        direct_velocity=800, dt=0.00025, mute_time0=0.010, 
                        dx=2, taper=20):
        """预处理单炮数据"""
        processed = self.normalize_traces_per_trace(shot_gather)
        
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
                     direct_velocity=800, mute_time0=0.010, dx=2, dt=0.00025):
        """对单炮数据进行去噪（支持手动指定能量增强系数）"""
        input_tensor = torch.from_numpy(shot_gather).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        output = self.model(input_tensor)
        denoised = output.squeeze().cpu().numpy()
        
        time_samples, trace_num = denoised.shape
        
        direct_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        
        if boost_sw_energy and sw_mask is not None:
            print(f"\n应用面波能量增强（手动系数: {energy_boost_factor:.2f}x）...")
            denoised = self.boost_surface_wave_energy( 
                denoised, 
                sw_mask,
                shot_center,
                energy_boost_factor, 
                direct_velocity,
                mute_time0,
                dx,
                dt
            )
        
        if replace_non_sw_with_input and sw_mask is not None:
            valid_sw_mask = sw_mask & direct_mask
            non_sw_mask = direct_mask & (~valid_sw_mask)
            
            denoised[non_sw_mask] = shot_gather[non_sw_mask]
            
            sw_pixels = valid_sw_mask.sum()
            non_sw_pixels = non_sw_mask.sum()
            
            print(f"\n拼接结果:")
            if boost_sw_energy:
                print(f"  面波区域: {sw_pixels} 个像素 → 网络输出 × {energy_boost_factor:.2f}")
            else:
                print(f"  面波区域: {sw_pixels} 个像素 → 网络输出（原始）")
            print(f"  非面波区域: {non_sw_pixels} 个像素 → 输入数据")
        
        denoised[~direct_mask] = 0
        direct_zero_pixels = (~direct_mask).sum()
        print(f"  直达波以上区域: {direct_zero_pixels} 个像素 → 置0")
        
        return denoised, None
    
    def save_shot_to_npy(self, data, output_path):
        """保存单炮数据为numpy格式"""
        np.save(output_path, data)
        print(f"结果已保存到: {output_path}")
    
    def plot_comparison(self, original, network_input, denoised, sw_mask, 
                        shot_idx, shot_center=22, 
                        direct_velocity=800, mute_time0=0.010, dx=2, dt=0.00025,
                        input_with_mask=True, boost_enabled=False,
                        energy_boost_factor=1.0, velocity_filter_enabled=False,
                        sw_min_velocity=200, sw_max_velocity=800,
                        save_path=None, figsize=(18, 12)):
        
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        time_samples, trace_num = original.shape
        direct_wave_mask = self.get_direct_wave_mask(
            time_samples, trace_num, shot_center,
            direct_velocity, mute_time0, dx, dt
        )
        vmax = np.percentile(np.abs(original), 99)
        
        # ========== 第一行 ==========
        
        # 1. 预处理后的原始数据
        im0 = axes[0, 0].imshow(original, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0, 0].set_title(f'1. 预处理后原始数据 - 第{shot_idx}炮', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('道号')
        axes[0, 0].set_ylabel('时间采样点')
        axes[0, 0].axvline(shot_center, color='lime', linestyle='--', linewidth=2, label='炮点')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 面波区域标识（带速度约束标注）
        sw_mask_display = sw_mask.copy()
        sw_mask_display[~direct_wave_mask] = 0
        im1 = axes[0, 1].imshow(sw_mask_display, aspect='auto', cmap='RdYlGn_r',
                                vmin=0, vmax=1, interpolation='nearest')
        title_str = '2. 面波区域标识'
        if velocity_filter_enabled:
            title_str += f'\n(速度约束: {sw_min_velocity}-{sw_max_velocity} m/s)'
        axes[0, 1].set_title(title_str, fontsize=10)
        axes[0, 1].set_xlabel('道号')
        axes[0, 1].set_ylabel('时间采样点')
        axes[0, 1].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. 网络输入
        im2 = axes[0, 2].imshow(network_input, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        title_str = f'3. 网络输入（面波→掩码）' if input_with_mask else '3. 网络输入（原始）'
        axes[0, 2].set_title(title_str, fontsize=11, fontweight='bold')
        axes[0, 2].set_xlabel('道号')
        axes[0, 2].set_ylabel('时间采样点')
        axes[0, 2].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im2, ax=axes[0, 2])
        
        # ========== 第二行 ==========
        
        # 4. 网络输入 + 面波区域叠加
        axes[1, 0].imshow(network_input, aspect='auto', cmap='gray',
                          vmin=-vmax, vmax=vmax, interpolation='bilinear')
        masked_sw = np.ma.masked_where(sw_mask_display == 0, sw_mask_display)
        im3 = axes[1, 0].imshow(masked_sw, aspect='auto', cmap='Reds', 
                               alpha=0.4, vmin=0, vmax=1, interpolation='nearest')
        axes[1, 0].set_title('4. 网络输入 + 面波区域叠加', fontsize=11)
        axes[1, 0].set_xlabel('道号')
        axes[1, 0].set_ylabel('时间采样点')
        axes[1, 0].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        
        # 5. 最终结果
        im5 = axes[1, 1].imshow(denoised, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        if boost_enabled:
            result_title = f'5. 最终结果（增强×{energy_boost_factor:.2f}+拼接+直达波置0）'
        else:
            result_title = '5. 最终结果（直接拼接+直达波置0）'
        axes[1, 1].set_title(result_title, fontsize=11, fontweight='bold', color='green')
        axes[1, 1].set_xlabel('道号')
        axes[1, 1].set_ylabel('时间采样点')
        axes[1, 1].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im5, ax=axes[1, 1])
        
        # 6. 总体差异
        total_diff = original - denoised
        vmax_diff = np.percentile(np.abs(total_diff), 99)
        im7 = axes[1, 2].imshow(total_diff, aspect='auto', cmap='seismic',
                                vmin=-vmax_diff, vmax=vmax_diff, interpolation='bilinear')
        axes[1, 2].set_title(f'6. 总体差异（原始 - 最终）', fontsize=11)
        axes[1, 2].set_xlabel('道号')
        axes[1, 2].set_ylabel('时间采样点')
        axes[1, 2].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im7, ax=axes[1, 2])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存: {save_path}")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
        
    # ========== 配置参数 ==========
    model_path = r"D:\桌面\面波压制\model\model_syn_1027\model_epoch_500.pth"
    input_segy_path = r"D:\桌面\面波压制\data\cs_free.sgy"
    output_dir = r"D:\桌面\面波压制\方法对比\本文方法\with_direct3"
    os.makedirs(output_dir, exist_ok=True)

    USE_MASK_FOR_SURFACE_WAVE = True  # 是否在输入时替换面波为掩码
    BOOST_SURFACE_WAVE_ENERGY = True   # 是否进行面波能量增强
    ENERGY_BOOST_FACTOR = 2          # 手动指定能量增强系数（建议1.0-3.0）
    APPLY_VELOCITY_CONSTRAINT = False   # 是否应用速度约束
    SW_MIN_VELOCITY = 50              # 面波最小速度 (m/s)
    SW_MAX_VELOCITY = 170              # 面波最大速度 (m/s)
    

    target_shot_idx = 50            # 炮号
    shot_center = 24                # 炮点位置
    shotnum = 260                   # 炮集数量
    max_time_samples = 1200         # 显示时长 
    remove_direct = True            # 直达波切除
    direct_velocity = 1200          # 直达波速度 
    dt = 0.00025                    # 采样间隔
    mute_time0 = 0.025              # 切除起始时间
    dx = 2                          # 道间距 
    taper = 20                      # 切除过渡带宽度 
    window_width = 0.015            # 窗宽
    time_shift = 0.001              # 步长  
    sw_threshold = 0.08              # 面波识别阈值
    fixed_mask_value = 0.5          # 高斯掩码均值 
    
    print("=" * 70)
    print("地震数据单炮去噪推理（带速度约束）")
    print("=" * 70)
    print(f"输入模式: {'面波替换为掩码' if USE_MASK_FOR_SURFACE_WAVE else '面波保持原始数据'}")
    print(f"能量增强: {'启用' if BOOST_SURFACE_WAVE_ENERGY else '禁用'}")
    if BOOST_SURFACE_WAVE_ENERGY:
        print(f"增强系数: {ENERGY_BOOST_FACTOR:.2f}x")
    print(f"速度约束: {'启用' if APPLY_VELOCITY_CONSTRAINT else '禁用'}")
    if APPLY_VELOCITY_CONSTRAINT:
        print(f"速度范围: {SW_MIN_VELOCITY} - {SW_MAX_VELOCITY} m/s")
    print(f"⚠️ 重要提示: 直达波以上区域将被强制置0")
    print("=" * 70)
    
    denoiser = SeismicDenoiser(
        model_path,
        window_width=window_width,
        time_shift=time_shift,
        sw_threshold=sw_threshold,
        fixed_mask_value=fixed_mask_value
    )
    
    # ========== 读取数据 ==========
    print("\n" + "=" * 70)
    print("读取SEGY数据...")
    print("=" * 70)
    seismic_data = denoiser.read_segy(input_segy_path, shotnum=shotnum)
    seismic_data = seismic_data[:, :max_time_samples, :]
    print(f"数据形状: {seismic_data.shape}")
    
    if target_shot_idx >= seismic_data.shape[0] or target_shot_idx < 0:
        print(f"错误: 炮号 {target_shot_idx} 超出范围 [0, {seismic_data.shape[0]-1}]")
        exit(1)
    
    # ========== 提取单炮数据 ==========
    print(f"\n提取第 {target_shot_idx} 炮数据...")
    single_shot = seismic_data[target_shot_idx]
    print(f"单炮形状: {single_shot.shape}")
    
    # ========== 预处理 ==========
    print("\n" + "=" * 70)
    print("预处理数据...")
    print("=" * 70)
    
    preprocessed_data = denoiser.preprocess_shot(
        single_shot,
        shot_center=shot_center,
        remove_direct=remove_direct,
        direct_velocity=direct_velocity,
        dt=dt,
        mute_time0=mute_time0,
        dx=dx,
        taper=taper
    )
    print("基础预处理完成!")
    
    # ========== 识别面波区域（带速度约束） ==========
    print("\n" + "=" * 70)
    print("识别面波区域...")
    if APPLY_VELOCITY_CONSTRAINT:
        print(f"启用速度约束: {SW_MIN_VELOCITY} - {SW_MAX_VELOCITY} m/s")
    print("=" * 70)
    
    sw_mask = denoiser.identify_surface_wave(
        preprocessed_data,
        dt=dt,
        shot_center=shot_center,
        direct_velocity=direct_velocity,
        mute_time0=mute_time0,
        dx=dx,
        apply_velocity_filter=APPLY_VELOCITY_CONSTRAINT,
        sw_min_velocity=SW_MIN_VELOCITY,
        sw_max_velocity=SW_MAX_VELOCITY
    )
    
    # 准备网络输入
    if USE_MASK_FOR_SURFACE_WAVE:
        print(f"\n将面波区域替换为掩码值 {fixed_mask_value}...")
        network_input = denoiser.replace_surface_wave_with_mask(
            preprocessed_data,
            sw_mask=sw_mask,
            shot_center=shot_center,
            direct_velocity=direct_velocity,
            mute_time0=mute_time0,
            dx=dx,
            dt=dt
        )
    else:
        print("\n保持面波区域为原始数据...")
        network_input = preprocessed_data.copy()
    
    # ========== 执行去噪（带能量增强） ==========
    print("\n" + "=" * 70)
    print("开始去噪...")
    print("=" * 70)
    
    denoised_shot, _ = denoiser.denoise_shot(
        network_input,
        sw_mask=sw_mask,
        shot_center=shot_center,
        replace_non_sw_with_input=True,
        boost_sw_energy=BOOST_SURFACE_WAVE_ENERGY,
        energy_boost_factor=ENERGY_BOOST_FACTOR,
        direct_velocity=direct_velocity,
        mute_time0=mute_time0,
        dx=dx,
        dt=dt
    )
    print("去噪完成!")
    print(f"去噪后数据形状: {denoised_shot.shape}")
    
    # ========== 保存结果 ==========
    print("\n" + "=" * 70)
    print("保存去噪结果...")
    print("=" * 70)
    
    mode_suffix = "with_mask" if USE_MASK_FOR_SURFACE_WAVE else "no_mask"
    boost_suffix = f"_boost{ENERGY_BOOST_FACTOR:.2f}" if BOOST_SURFACE_WAVE_ENERGY else "_no_boost"
    velocity_suffix = f"_vel{SW_MIN_VELOCITY}-{SW_MAX_VELOCITY}" if APPLY_VELOCITY_CONSTRAINT else ""
    
    # 保存去噪后的数据
    output_npy_path = os.path.join(output_dir, f"denoised_shot_{target_shot_idx}_{mode_suffix}{boost_suffix}{velocity_suffix}.npy")
    denoiser.save_shot_to_npy(denoised_shot, output_npy_path)
    
    # 保存预处理后的原始数据
    original_npy_path = os.path.join(output_dir, f"original_shot_{target_shot_idx}.npy")
    denoiser.save_shot_to_npy(preprocessed_data, original_npy_path)
    print(f"预处理后原始数据已保存: {original_npy_path}")
    
    # 保存网络输入数据
    input_npy_path = os.path.join(output_dir, f"input_shot_{target_shot_idx}_{mode_suffix}{velocity_suffix}.npy")
    denoiser.save_shot_to_npy(network_input, input_npy_path)
    print(f"网络输入数据已保存: {input_npy_path}")
    
    # 保存面波掩码
    sw_mask_npy_path = os.path.join(output_dir, f"sw_mask_shot_{target_shot_idx}{velocity_suffix}.npy")
    denoiser.save_shot_to_npy(sw_mask, sw_mask_npy_path)
    print(f"面波掩码已保存: {sw_mask_npy_path}")

    # 保存噪声数据（原始 - 去噪）
    noise_data = preprocessed_data - denoised_shot
    noise_npy_path = os.path.join(output_dir, f"noise_shot_{target_shot_idx}_{mode_suffix}{boost_suffix}{velocity_suffix}.npy")
    denoiser.save_shot_to_npy(noise_data, noise_npy_path)
    
    # ========== 绘制对比图 ==========
    print("\n" + "=" * 70)
    print("生成对比图...")
    print("=" * 70)
    
    comparison_path = os.path.join(output_dir, f"comparison_shot_{target_shot_idx}_{mode_suffix}{boost_suffix}{velocity_suffix}.png")
    denoiser.plot_comparison(
        preprocessed_data,
        network_input,
        denoised_shot,
        sw_mask,
        target_shot_idx,
        shot_center=shot_center,
        direct_velocity=direct_velocity,
        mute_time0=mute_time0,
        dx=dx,
        dt=dt,
        input_with_mask=USE_MASK_FOR_SURFACE_WAVE,
        boost_enabled=BOOST_SURFACE_WAVE_ENERGY,
        energy_boost_factor=ENERGY_BOOST_FACTOR,
        velocity_filter_enabled=APPLY_VELOCITY_CONSTRAINT,
        sw_min_velocity=SW_MIN_VELOCITY,
        sw_max_velocity=SW_MAX_VELOCITY,
        save_path=comparison_path
    )
    
    print("\n显示对比图...")
    denoiser.plot_comparison(
        preprocessed_data,
        network_input,
        denoised_shot,
        sw_mask,
        target_shot_idx,
        shot_center=shot_center,
        direct_velocity=direct_velocity,
        mute_time0=mute_time0,
        dx=dx,
        dt=dt,
        input_with_mask=USE_MASK_FOR_SURFACE_WAVE,
        boost_enabled=BOOST_SURFACE_WAVE_ENERGY,
        energy_boost_factor=ENERGY_BOOST_FACTOR,
        velocity_filter_enabled=APPLY_VELOCITY_CONSTRAINT,
        sw_min_velocity=SW_MIN_VELOCITY,
        sw_max_velocity=SW_MAX_VELOCITY,
        save_path=None
    )
    
    # ========== 完成 ==========
    print("\n" + "=" * 70)
    print("推理完成！")
    print("=" * 70)
    print(f"处理的炮号: {target_shot_idx}")
    print(f"输入模式: {'面波替换为掩码' if USE_MASK_FOR_SURFACE_WAVE else '面波保持原始数据'}")
    print(f"能量增强: {'启用（系数=' + str(ENERGY_BOOST_FACTOR) + 'x）' if BOOST_SURFACE_WAVE_ENERGY else '禁用'}")
    print(f"速度约束: {'启用（' + str(SW_MIN_VELOCITY) + '-' + str(SW_MAX_VELOCITY) + ' m/s）' if APPLY_VELOCITY_CONSTRAINT else '禁用'}")
    print(f"\n保存的文件:")
    print(f"- 预处理后原始数据: {original_npy_path}")
    print(f"- 网络输入数据: {input_npy_path}")
    print(f"- 去噪后数据: {output_npy_path}")
    print(f"- 面波区域掩码: {sw_mask_npy_path}")
    print(f"- 对比图: {comparison_path}")
    print("\n处理流程总结:")
    print(f"1. 读取原始数据")
    print(f"2. 逐道归一化")
    print(f"3. 去除直达波")
    print(f"4. 基于能量阈值识别面波区域（阈值={sw_threshold}）")
    if APPLY_VELOCITY_CONSTRAINT:
        print(f"5. 🆕 应用速度约束过滤（{SW_MIN_VELOCITY}-{SW_MAX_VELOCITY} m/s）")
        step_num = 6
    else:
        step_num = 5
    if USE_MASK_FOR_SURFACE_WAVE:
        print(f"{step_num}. 将面波区域替换为掩码值 {fixed_mask_value}")
    else:
        print(f"{step_num}. 保持面波区域为原始数据")
    step_num += 1
    print(f"{step_num}. 输入网络进行推理")
    step_num += 1
    if BOOST_SURFACE_WAVE_ENERGY:
        print(f"{step_num}. 对面波区域进行能量增强（系数={ENERGY_BOOST_FACTOR}x）")
        step_num += 1
        print(f"{step_num}. 非面波区域用输入数据替换")
        step_num += 1
        print(f"{step_num}. ⚠️ 直达波以上区域强制置0")
        print(f"\n最终结果：面波（网络输出×{ENERGY_BOOST_FACTOR}）+ 非面波（输入数据）+ 直达波以上（0）")
    else:
        print(f"{step_num}. 非面波区域用输入数据替换")
        step_num += 1
        print(f"{step_num}. ⚠️ 直达波以上区域强制置0")
        print(f"\n最终结果：面波（网络输出）+ 非面波（输入数据）+ 直达波以上（0）")
    print("=" * 70)