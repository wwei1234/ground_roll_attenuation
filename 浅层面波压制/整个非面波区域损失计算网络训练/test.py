import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
from U_Net_CBAM import UNet

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


class SeismicDenoiser:
    """地震数据去噪推理器（带面波掩码）"""
    
    def __init__(self, model_path, device=None):
        """
        初始化推理器
        
        Args:
            model_path: 训练好的模型权重路径
            device: 推理设备，默认自动选择
        """
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
    
    def get_direct_wave_mask(self, shot_shape, center_trace, velocity, mute_time0, dx, dt):
        """
        计算直达波区域掩码
        返回: 布尔数组，True表示在直达波以下（可以添加掩码的区域）
        """
        time_samples, trace_num = shot_shape
        direct_mask = np.zeros((time_samples, trace_num), dtype=bool)
        
        for tr in range(trace_num):
            offset = abs(tr - center_trace)*dx
            t0 = mute_time0 + offset / velocity
            t0_idx = int(t0/dt)
            
            # 直达波以下的区域可以添加掩码
            if t0_idx < time_samples:
                direct_mask[t0_idx:, tr] = True
        
        return direct_mask
    
    def get_line_pixels(self, point_trace, point_time, shot_center, time_samples, trace_num):
        """获取从活跃像素沿炮点连线方向反向（向下）延伸的所有像素位置"""
        pixels = []
        t0, x0 = 0, shot_center        # 炮点坐标（假定在顶部）
        t1, x1 = point_time, point_trace  # 活跃像素坐标

        dt = t1 - t0
        dx = x1 - x0

        # 若 dt=0，则避免除零错误
        if dt == 0 and dx == 0:
            return [(t1, x1)]

        # 计算连线方向（炮点→活跃像素）
        length = np.hypot(dx, dt)
        dx /= length
        dt /= length

        # 反向延伸方向（活跃像素→底部）
        if dt <= 0:  
            # 若射线原本朝上，则反转方向
            dx = -dx
            dt = abs(dt)

        x, t = x1, t1

        # 向下延伸直到图像底部
        while 0 <= int(x) < trace_num and 0 <= int(t) < time_samples:
            pixels.append((int(round(t)), int(round(x))))
            x += dx
            t += dt

        return pixels
    
    def apply_surface_wave_mask(self, shot_gather, sw_mask, shot_center, 
                                direct_velocity, mute_time0, dx, dt, 
                                density_range=(25, 35)):
        """
        在面波区域添加掩码
        
        Args:
            shot_gather: 预处理后的炮集数据 [time, traces]
            sw_mask: 面波识别掩码 [time, traces]
            shot_center: 炮点位置
            其他参数: 直达波去除参数
            density_range: 面波区域活跃像素密度范围 (每100像素)
        
        Returns:
            masked_data: 添加掩码后的数据
            mask_positions: 掩码位置 (bool数组)
        """
        time_samples, trace_num = shot_gather.shape
        masked_data = shot_gather.copy()
        mask_positions = np.zeros((time_samples, trace_num), dtype=bool)
        
        # 获取直达波以下的有效区域
        direct_wave_mask = self.get_direct_wave_mask(
            shot_gather.shape, shot_center, direct_velocity, mute_time0, dx, dt
        )
        
        # 只在直达波以下的面波区域添加掩码
        valid_sw_mask = sw_mask & direct_wave_mask
        
        # 获取有效区域的所有道
        valid_traces = np.where(np.any(valid_sw_mask, axis=0))[0]
        
        if len(valid_traces) == 0:
            print("警告: 没有检测到有效的面波区域!")
            return masked_data, mask_positions
        
        # 计算每道需要的活跃像素数量（基于密度）
        for trace_idx in valid_traces:
            # 该道的面波区域（直达波以下）
            sw_pixels_in_trace = np.where(valid_sw_mask[:, trace_idx])[0]
            
            # 面波区域：随机选择活跃像素
            if len(sw_pixels_in_trace) > 0:
                sw_density = np.random.uniform(*density_range)
                n_active_sw = max(1, int(len(sw_pixels_in_trace) * sw_density / 100))
                n_active_sw = min(n_active_sw, len(sw_pixels_in_trace))
                
                # 随机选择若干个时间采样点
                selected_times = np.random.choice(
                    sw_pixels_in_trace, 
                    size=n_active_sw, 
                    replace=False
                )
                
                for time_idx in selected_times:
                    # 从该像素到炮点画线
                    line_pixels = self.get_line_pixels(
                        trace_idx, time_idx, shot_center, time_samples, trace_num
                    )
                    
                    for t, x in line_pixels:
                        if (0 <= t < time_samples and 
                            0 <= x < trace_num and
                            direct_wave_mask[t, x]):  # 确保在直达波以下
                            std = np.std(shot_gather)
                            mean = np.mean(shot_gather)
                            masked_data[t, x] = np.random.normal(mean, std)
                            mask_positions[t, x] = True
        
        return masked_data, mask_positions
    
    def preprocess_shot(self, shot_gather, shot_center=22, remove_direct=True,
                       direct_velocity=800, dt=0.00025, mute_time0=0.010, 
                       dx=2, taper=20):
        """预处理单炮数据"""
        # 归一化
        processed = self.normalize_traces_per_trace(shot_gather)
        
        # 去除直达波
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
    def denoise_shot(self, shot_gather):
        """
        对单炮数据进行去噪
        
        Args:
            shot_gather: 输入的炮集数据 [H, W]
        
        Returns:
            denoised: 去噪后的数据 [H, W]
        """
        # 转换为tensor并添加batch和channel维度
        input_tensor = torch.from_numpy(shot_gather).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # 推理
        output = self.model(input_tensor)
        
        # 转回numpy
        denoised = output.squeeze().cpu().numpy()
        
        return denoised
    
    def save_shot_to_npy(self, data, output_path):
        """保存单炮数据为numpy格式"""
        np.save(output_path, data)
        print(f"结果已保存到: {output_path}")
    
    def plot_comparison(self, original, masked, denoised, mask_positions, sw_mask,
                       shot_idx, shot_center=22, save_path=None, figsize=(20, 12)):
        """
        绘制去噪前后对比图（包含掩码信息）
        
        Args:
            original: 预处理后的原始数据 [time, traces]
            masked: 添加掩码后的数据 [time, traces]
            denoised: 去噪后数据 [time, traces]
            mask_positions: 掩码位置 [time, traces]
            sw_mask: 面波识别掩码 [time, traces]
            shot_idx: 炮号
            shot_center: 炮点位置
            save_path: 保存路径，如果为None则显示
            figsize: 图像尺寸
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 计算显示范围
        vmax = np.percentile(np.abs(original), 99)
        
        # 第一行
        # 1. 预处理后原始数据
        im0 = axes[0, 0].imshow(original, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0, 0].set_title(f'预处理后原始数据 - 第{shot_idx}炮', fontsize=12)
        axes[0, 0].set_xlabel('道号')
        axes[0, 0].set_ylabel('时间采样点')
        axes[0, 0].axvline(shot_center, color='lime', linestyle='--', 
                          linewidth=2, label='炮点')
        axes[0, 0].legend(loc='upper right')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 面波识别区域
        im1 = axes[0, 1].imshow(sw_mask, aspect='auto', cmap='RdYlGn_r',
                                vmin=0, vmax=1, interpolation='nearest')
        axes[0, 1].set_title(f'识别的面波区域', fontsize=12)
        axes[0, 1].set_xlabel('道号')
        axes[0, 1].set_ylabel('时间采样点')
        axes[0, 1].axvline(shot_center, color='lime', linestyle='--', 
                          linewidth=2, label='炮点')
        axes[0, 1].legend(loc='upper right')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. 掩码位置
        im2 = axes[0, 2].imshow(mask_positions.astype(int), aspect='auto', 
                                cmap='RdYlGn_r', vmin=0, vmax=1, 
                                interpolation='nearest')
        axes[0, 2].set_title(f'掩码位置', fontsize=12)
        axes[0, 2].set_xlabel('道号')
        axes[0, 2].set_ylabel('时间采样点')
        axes[0, 2].axvline(shot_center, color='lime', linestyle='--', 
                          linewidth=2, label='炮点')
        axes[0, 2].legend(loc='upper right')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 第二行
        # 4. 添加掩码后的数据（网络输入）
        im3 = axes[1, 0].imshow(masked, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[1, 0].set_title(f'掩码数据 (网络输入)', fontsize=12)
        axes[1, 0].set_xlabel('道号')
        axes[1, 0].set_ylabel('时间采样点')
        axes[1, 0].axvline(shot_center, color='lime', linestyle='--', 
                          linewidth=2, label='炮点')
        axes[1, 0].legend(loc='upper right')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 5. 去噪后数据
        im4 = axes[1, 1].imshow(denoised, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[1, 1].set_title(f'去噪后数据 (网络输出)', fontsize=12)
        axes[1, 1].set_xlabel('道号')
        axes[1, 1].set_ylabel('时间采样点')
        axes[1, 1].axvline(shot_center, color='lime', linestyle='--', 
                          linewidth=2, label='炮点')
        axes[1, 1].legend(loc='upper right')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # 6. 去除的噪声
        noise = original - denoised
        vmax_noise = np.percentile(np.abs(noise), 99)
        im5 = axes[1, 2].imshow(noise, aspect='auto', cmap='seismic',
                                vmin=-vmax_noise, vmax=vmax_noise, interpolation='bilinear')
        axes[1, 2].set_title(f'去除的噪声', fontsize=12)
        axes[1, 2].set_xlabel('道号')
        axes[1, 2].set_ylabel('时间采样点')
        axes[1, 2].axvline(shot_center, color='lime', linestyle='--', 
                          linewidth=2, label='炮点')
        axes[1, 2].legend(loc='upper right')
        plt.colorbar(im5, ax=axes[1, 2])
        
        # 统计信息
        sw_ratio = np.sum(sw_mask) / sw_mask.size * 100
        mask_ratio = np.sum(mask_positions) / mask_positions.size * 100
        
        info_text = f'面波区域占比: {sw_ratio:.2f}% | 掩码覆盖率: {mask_ratio:.2f}%'
        fig.suptitle(info_text, fontsize=14, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存: {save_path}")
            plt.close()
        else:
            plt.show()


# ========== 使用示例 ==========
if __name__ == "__main__":
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # ========== 配置参数 ==========
    # 模型路径
    model_path = r"D:\桌面\面波压制\saved_models_2\model_epoch_300.pth"
    
    # 输入SEGY文件路径
    input_segy_path = r"D:\桌面\面波压制\data\f20_z_test+GathEP.sgy"
    
    # 输出路径
    output_dir = "denoised_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择要处理的炮号
    target_shot_idx = 100
    
    # 数据参数（需要与训练时保持一致）
    shot_center = 22
    shotnum = 700
    max_time_samples = 1200
    
    # 预处理参数（需要与训练时保持一致）
    remove_direct = True
    direct_velocity = 800
    dt = 0.00025
    mute_time0 = 0.010
    dx = 2
    taper = 20
    
    # 面波识别参数（需要与训练时保持一致）
    window_width = 0.01
    time_shift = 0.002
    sw_threshold = 0.08
    
    # 面波区域掩码密度（需要与训练时保持一致）
    sw_density_range = (0.00001, 0.00005)  # 每100像素中的活跃像素数量
    
    # ========== 初始化推理器和面波分析器 ==========
    print("=" * 60)
    print("地震数据单炮去噪推理 (面波区域掩码)")
    print("=" * 60)
    
    denoiser = SeismicDenoiser(model_path)
    analyzer = SeismicEnergyAnalyzer(window_width=window_width, time_shift=time_shift)
    
    # ========== 读取数据 ==========
    print("\n" + "=" * 60)
    print("读取SEGY数据...")
    print("=" * 60)
    seismic_data = denoiser.read_segy(input_segy_path, shotnum=shotnum)
    
    # 只保留前max_time_samples个采样点
    seismic_data = seismic_data[:, :max_time_samples, :]
    print(f"数据形状: {seismic_data.shape}")
    
    # 检查炮号是否有效
    if target_shot_idx >= seismic_data.shape[0] or target_shot_idx < 0:
        print(f"错误: 炮号 {target_shot_idx} 超出范围 [0, {seismic_data.shape[0]-1}]")
        exit(1)
    
    # ========== 提取单炮数据 ==========
    print(f"\n提取第 {target_shot_idx} 炮数据...")
    single_shot = seismic_data[target_shot_idx]
    print(f"单炮形状: {single_shot.shape}")
    
    # ========== 预处理 ==========
    print("\n" + "=" * 60)
    print("预处理数据...")
    print("=" * 60)
    
    preprocessed_shot = denoiser.preprocess_shot(
        single_shot,
        shot_center=shot_center,
        remove_direct=remove_direct,
        direct_velocity=direct_velocity,
        dt=dt,
        mute_time0=mute_time0,
        dx=dx,
        taper=taper
    )
    print("预处理完成!")
    
    # ========== 识别面波区域 ==========
    print("\n" + "=" * 60)
    print("识别面波区域...")
    print("=" * 60)
    
    sw_mask = analyzer.batch_identify_surface_wave(
        preprocessed_shot, 
        dt, 
        sw_threshold
    )
    sw_ratio = np.sum(sw_mask) / sw_mask.size * 100
    print(f"面波区域占比: {sw_ratio:.2f}%")
    
    # ========== 在面波区域添加掩码 ==========
    print("\n" + "=" * 60)
    print("在面波区域添加掩码...")
    print("=" * 60)
    
    masked_shot, mask_positions = denoiser.apply_surface_wave_mask(
        preprocessed_shot,
        sw_mask,
        shot_center=shot_center,
        direct_velocity=direct_velocity,
        mute_time0=mute_time0,
        dx=dx,
        dt=dt,
        density_range=sw_density_range
    )
    mask_ratio = np.sum(mask_positions) / mask_positions.size * 100
    print(f"掩码覆盖率: {mask_ratio:.2f}%")
    
    # ========== 执行去噪 ==========
    print("\n" + "=" * 60)
    print("开始去噪...")
    print("=" * 60)
    
    denoised_shot = denoiser.denoise_shot(masked_shot)
    print("去噪完成!")
    print(f"去噪后数据形状: {denoised_shot.shape}")
    
    # ========== 保存结果 ==========
    print("\n" + "=" * 60)
    print("保存去噪结果...")
    print("=" * 60)
    
    # 保存各个阶段的数据
    denoiser.save_shot_to_npy(
        preprocessed_shot, 
        os.path.join(output_dir, f"original_shot_{target_shot_idx}.npy")
    )
    denoiser.save_shot_to_npy(
        masked_shot, 
        os.path.join(output_dir, f"masked_shot_{target_shot_idx}.npy")
    )
    denoiser.save_shot_to_npy(
        denoised_shot, 
        os.path.join(output_dir, f"denoised_shot_{target_shot_idx}.npy")
    )
    denoiser.save_shot_to_npy(
        sw_mask.astype(np.float32), 
        os.path.join(output_dir, f"sw_mask_{target_shot_idx}.npy")
    )
    denoiser.save_shot_to_npy(
        mask_positions.astype(np.float32), 
        os.path.join(output_dir, f"mask_positions_{target_shot_idx}.npy")
    )
    
    # ========== 绘制对比图 ==========
    print("\n" + "=" * 60)
    print("生成对比图...")
    print("=" * 60)
    
    comparison_path = os.path.join(output_dir, f"comparison_shot_{target_shot_idx}.png")
    denoiser.plot_comparison(
        preprocessed_shot,
        masked_shot,
        denoised_shot,
        mask_positions,
        sw_mask,
        target_shot_idx,
        shot_center=shot_center,
        save_path=comparison_path
    )
    
    # 也显示在屏幕上
    print("\n显示对比图...")
    denoiser.plot_comparison(
        preprocessed_shot,
        masked_shot,
        denoised_shot,
        mask_positions,
        sw_mask,
        target_shot_idx,
        shot_center=shot_center,
        save_path=None
    )
    
    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    print(f"处理的炮号: {target_shot_idx}")
    print(f"面波区域占比: {sw_ratio:.2f}%")
    print(f"掩码覆盖率: {mask_ratio:.2f}%")
    print(f"\n保存的文件:")
    print(f"  - 预处理后原始数据: original_shot_{target_shot_idx}.npy")
    print(f"  - 掩码数据 (网络输入): masked_shot_{target_shot_idx}.npy")
    print(f"  - 去噪后数据 (网络输出): denoised_shot_{target_shot_idx}.npy")
    print(f"  - 面波识别掩码: sw_mask_{target_shot_idx}.npy")
    print(f"  - 掩码位置: mask_positions_{target_shot_idx}.npy")
    print(f"  - 对比图: comparison_shot_{target_shot_idx}.png")
    print("=" * 60)