import numpy as np
import matplotlib.pyplot as plt
import segyio
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class SeismicEnergyAnalyzer:
    """地震数据能量片段分析器"""
    
    def __init__(self, window_width=0.02, time_shift=0.01):
        """
        初始化分析器
        
        参数:
        window_width: 高斯窗函数宽度，单位秒 (论文中使用0.08s)
        time_shift: 每次移动的时间长度，单位秒 (论文中使用0.01s)
        """
        self.window_width = window_width
        self.time_shift = time_shift
        
    def gaussian_window(self, t, center, alpha):
        """
        生成归一化高斯窗函数
        
        参数:
        t: 时间数组
        center: 窗中心位置
        alpha: 窗宽度参数 (alpha = 1/window_width)
        
        返回:
        归一化的高斯窗函数
        """
        X = np.sqrt(alpha / np.pi)  # 归一化参数
        window = X * np.exp(-alpha**2 * (t - center)**2)
        return window
    
    def compute_segment_energy(self, seismic_trace, dt):
        """
        计算地震道每个片段的平均振幅能量
        参数:
        seismic_trace: 地震道数据 (1D numpy数组)
        dt: 采样间隔，单位秒
        
        返回:
        time_centers: 各片段的中心时间
        energies: 各片段的平均振幅能量
        segments: 各个地震道片段
        """
        n_samples = len(seismic_trace)
        t = np.arange(n_samples) * dt
        
        # 计算窗宽度参数
        alpha = 1.0 / self.window_width
        
        # 计算时窗中心位置
        n_windows = int((t[-1] - t[0]) / self.time_shift) + 1
        time_centers = np.arange(n_windows) * self.time_shift + t[0]
        
        energies = []
        segments = []
        
        for center in time_centers:
            # 生成高斯窗
            window = self.gaussian_window(t, center, alpha)
            
            # 应用窗函数得到地震道片段
            segment = seismic_trace * window
            segments.append(segment)
            
            # 计算该片段的平均振幅 (公式5)
            avg_amplitude = np.mean(np.abs(segment))
            energies.append(avg_amplitude)
        
        return time_centers, np.array(energies), segments
    
    def identify_surface_wave(self, energies, threshold=0.001):
        """
        根据能量阈值识别含有面波的时间段
        
        参数:
        energies: 各片段的能量值
        threshold: 能量阈值
        
        返回:
        surface_wave_mask: 布尔数组，True表示含有面波
        """
        return energies > threshold
    
    def plot_energy_statistics(self, time_centers, energies, threshold=0.001, 
                              title='地震道片段能量统计图'):
        """
        绘制能量统计图
        
        参数:
        time_centers: 时间中心数组
        energies: 能量数组
        threshold: 能量阈值
        title: 图标题
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制能量曲线
        plt.plot(time_centers, energies, 'b-', linewidth=2, label='片段能量')
        
        # 绘制阈值线
        plt.axhline(y=threshold, color='r', linestyle='--', 
                   linewidth=1.5, label=f'面波阈值 = {threshold}')
        
        # 标记面波区域
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        if np.any(surface_wave_mask):
            plt.fill_between(time_centers, 0, energies, 
                           where=surface_wave_mask, 
                           alpha=0.3, color='red', 
                           label='识别的面波区域')
        
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('平均振幅能量', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_trace_with_windows(self, seismic_trace, dt, time_centers, 
                               energies, threshold=0.001):
        """
        绘制地震道及其能量分析结果
        
        参数:
        seismic_trace: 地震道数据
        dt: 采样间隔
        time_centers: 时间中心
        energies: 能量值
        threshold: 阈值
        """
        t = np.arange(len(seismic_trace)) * dt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制地震道
        ax1.plot(t, seismic_trace, 'k-', linewidth=1)
        ax1.set_xlabel('时间 (s)', fontsize=12)
        ax1.set_ylabel('振幅', fontsize=12)
        ax1.set_title('地震道记录', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 标记面波时间段
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        if np.any(surface_wave_mask):
            surface_wave_times = time_centers[surface_wave_mask]
            for tw_time in surface_wave_times:
                ax1.axvspan(tw_time - self.window_width/2, 
                          tw_time + self.window_width/2,
                          alpha=0.2, color='red')
        
        # 绘制能量统计
        ax2.plot(time_centers, energies, 'b-', linewidth=2, label='片段能量')
        ax2.axhline(y=threshold, color='r', linestyle='--', 
                   linewidth=1.5, label=f'阈值 = {threshold}')
        ax2.fill_between(time_centers, 0, energies, 
                        where=surface_wave_mask, 
                        alpha=0.3, color='red', 
                        label='面波区域')
        ax2.set_xlabel('时间 (s)', fontsize=12)
        ax2.set_ylabel('平均振幅能量', fontsize=12)
        ax2.set_title('能量统计图', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_trace_mask(self, seismic_trace, dt, threshold=0.001):
        """
        为单道地震记录创建面波掩码
        
        参数:
        seismic_trace: 地震道数据
        dt: 采样间隔
        threshold: 能量阈值
        
        返回:
        mask: 面波掩码 (1=面波, 0=其他)
        """
        n_samples = len(seismic_trace)
        t = np.arange(n_samples) * dt
        
        # 计算能量统计
        time_centers, energies, _ = self.compute_segment_energy(seismic_trace, dt)
        
        # 识别面波片段
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        
        # 创建样本级别的掩码
        mask = np.zeros(n_samples, dtype=int)
        
        for i, center in enumerate(time_centers):
            if surface_wave_mask[i]:
                # 找到该窗口对应的时间范围
                t_start = center - self.window_width / 2
                t_end = center + self.window_width / 2
                
                # 将该时间范围内的样本标记为面波
                idx_start = int(t_start / dt)
                idx_end = int(t_end / dt)
                
                idx_start = max(0, idx_start)
                idx_end = min(n_samples, idx_end)
                
                mask[idx_start:idx_end] = 1
        
        return mask
    
    def batch_identify_surface_wave(self, shot_gather, dt, threshold=0.001):
        """
        批量识别炮集中每道的面波区域
        
        参数:
        shot_gather: 炮集数据 (2D numpy数组, 形状为 [时间样点数, 道数])
        dt: 采样间隔
        threshold: 能量阈值
        
        返回:
        mask_gather: 面波掩码矩阵 (形状与shot_gather相同, 1=面波, 0=其他)
        """
        n_samples, n_traces = shot_gather.shape
        mask_gather = np.zeros_like(shot_gather, dtype=int)
        
        print(f"开始批量识别面波区域...")
        print(f"炮集信息: {n_traces}道, {n_samples}个时间样点")
        
        for i in range(n_traces):
            # 逐道处理
            trace = shot_gather[:, i]
            mask = self.create_trace_mask(trace, dt, threshold)
            mask_gather[:, i] = mask
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{n_traces} 道")
        
        print("面波识别完成！")
        
        # 统计面波区域
        total_samples = mask_gather.size
        surface_wave_samples = np.sum(mask_gather)
        percentage = (surface_wave_samples / total_samples) * 100
        print(f"面波区域占比: {percentage:.2f}%")
        
        return mask_gather
    
    def plot_shot_gather_with_mask(self, shot_gather, mask_gather, dt, 
                                   clip_percentile=99, figsize=(14, 8)):
        """
        绘制炮集记录及其面波掩码
        
        参数:
        shot_gather: 炮集数据
        mask_gather: 面波掩码
        dt: 采样间隔
        clip_percentile: 振幅显示的百分位裁剪
        figsize: 图像大小
        """
        n_samples, n_traces = shot_gather.shape
        t = np.arange(n_samples) * dt
        traces = np.arange(n_traces)
        
        # 计算显示范围
        vmax = np.percentile(np.abs(shot_gather), clip_percentile)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. 原始炮集记录
        im1 = axes[0].imshow(shot_gather, aspect='auto', cmap='seismic',
                            extent=[0, n_traces-1, t[-1], t[0]],
                            vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0].set_xlabel('道号', fontsize=12)
        axes[0].set_ylabel('时间 (s)', fontsize=12)
        axes[0].set_title('原始炮集记录', fontsize=14)
        plt.colorbar(im1, ax=axes[0], label='振幅')
        
        # 2. 面波掩码
        im2 = axes[1].imshow(mask_gather, aspect='auto', cmap='RdYlGn_r',
                            extent=[0, n_traces-1, t[-1], t[0]],
                            vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_xlabel('道号', fontsize=12)
        axes[1].set_ylabel('时间 (s)', fontsize=12)
        axes[1].set_title('面波区域掩码 (1=面波, 0=其他)', fontsize=14)
        plt.colorbar(im2, ax=axes[1], label='掩码值')
        
        # 3. 叠加显示
        axes[2].imshow(shot_gather, aspect='auto', cmap='gray',
                      extent=[0, n_traces-1, t[-1], t[0]],
                      vmin=-vmax, vmax=vmax, interpolation='bilinear')
        # 叠加面波区域（半透明红色）
        masked_data = np.ma.masked_where(mask_gather == 0, mask_gather)
        axes[2].imshow(masked_data, aspect='auto', cmap='Reds',
                      extent=[0, n_traces-1, t[-1], t[0]],
                      alpha=0.4, vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_xlabel('道号', fontsize=12)
        axes[2].set_ylabel('时间 (s)', fontsize=12)
        axes[2].set_title('炮集 + 面波区域标注', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def save_mask_to_file(self, mask_gather, output_path):
        """
        保存面波掩码到文件
        
        参数:
        mask_gather: 面波掩码矩阵
        output_path: 输出文件路径 (.npy 或 .txt)
        """
        if output_path.endswith('.npy'):
            np.save(output_path, mask_gather)
            print(f"掩码已保存到: {output_path}")
        elif output_path.endswith('.txt'):
            np.savetxt(output_path, mask_gather, fmt='%d')
            print(f"掩码已保存到: {output_path}")
        else:
            raise ValueError("输出文件格式必须是 .npy 或 .txt")


def read_segy(data_dir, shotnum=0):
    """
    读取SEGY格式地震数据
    
    参数:
    data_dir: SEGY文件路径
    shotnum: 炮数（0表示自动计算）
    
    返回:
    data: 地震数据 [炮号, 时间, 道号]
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
        print(f'开始读取SEGY数据: {shot_num}炮, 每炮{len_shot}道, {time}个时间样点')
        data = np.zeros((shot_num, time, len_shot))
        for j in range(shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        print('SEGY数据读取完成')
        return data

def remove_direct_wave(data, center_trace=64.5, velocity=800, dt=0.00025, mute_time0=0.01, taper=20, dx = 2, plot=True):
    n_time, n_trace = data.shape
    muted_data = data.copy()
    mask = np.ones_like(data)

    # 计算每个道的直达波走时
    for tr in range(n_trace):
        offset = abs(tr - center_trace)*dx  # 支持小数中心
        t0 = mute_time0 + offset / velocity
        t0_idx = int(t0/dt)

        if t0_idx < n_time:
            mask[:t0_idx, tr] = 0
            taper_end = min(t0_idx + taper, n_time)
            for t in range(t0_idx, taper_end):
                mask[t, tr] = 1 - (t - t0_idx) / taper

    muted_data *= mask

    # 可视化结果
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        vmax = np.max(np.abs(data)) * 0.8
        axs[0].imshow(data, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
        axs[0].set_title('原始炮集')
        axs[1].imshow(muted_data, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
        axs[1].set_title('去除直达波后')
        for ax in axs:
            ax.set_xlabel('道号')
            ax.set_ylabel('时间采样点')
            ax.axvline(center_trace, color='lime', linestyle='--', lw=1.5, label='炮点')
        plt.tight_layout()
        plt.show()

    return muted_data

def normalize_traces_per_trace(shot_gather):
    """
    对炮集数据进行逐道归一化处理，使每道的最大绝对振幅为1。
    
    参数:
        shot_gather (ndarray): 形状为 [n_samples, n_traces] 的二维数组。
        
    返回:
        normalized_gather (ndarray): 归一化后的炮集数据。
    """
    # 复制输入，避免修改原数据
    normalized_gather = np.copy(shot_gather)
    
    # 获取道数
    n_samples, n_traces = normalized_gather.shape
    
    for i in range(n_traces):
        trace = normalized_gather[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:  # 避免除零错误
            normalized_gather[:, i] = trace / max_amp
    
    return normalized_gather

# 使用示例
if __name__ == "__main__":
    data_path = r"D:\桌面\面波压制\data\modified_SN03.sgy"
    seismic_data = read_segy(data_path, shotnum=168)
    
    # 提取第一炮的第32道
    dt = 0.00025  # 采样间隔 (秒)
    shot_gather = seismic_data[13][0:1200]  # 第一炮 [时间, 道号]
    # 创建分析器
    analyzer = SeismicEnergyAnalyzer(window_width=0.01, time_shift=0.002)
    shot_gather_normalized = normalize_traces_per_trace(shot_gather)
    shot_gather_normalized = remove_direct_wave(
    shot_gather_normalized,
    center_trace=26,   # 炮点位置
    velocity=1200,     # 控制直达波倾角，值越小，倾角越大
    dt=0.00025,          # 采样间隔4ms
    mute_time0=0.010,     # 炮点处直达波到时
    taper=20,          # 渐变区宽度
    dx=2,              # 道间距
    plot=False
)
    
    print(np.max(shot_gather_normalized))
    # 批量识别面波区域
    mask_gather = analyzer.batch_identify_surface_wave(
        shot_gather_normalized, 
        dt, 
        threshold=0.08
    )
    
    # 绘制炮集及面波掩码
    fig3 = analyzer.plot_shot_gather_with_mask(
        shot_gather_normalized, 
        mask_gather, 
        dt,
        clip_percentile=99
    )
    
    # 保存面波掩码
    analyzer.save_mask_to_file(mask_gather, 'surface_wave_mask.npy')
    analyzer.save_mask_to_file(mask_gather, 'surface_wave_mask.txt')
    
    # 统计每道的面波占比
    print("\n每道面波占比统计:")
    n_samples, n_traces = mask_gather.shape
    for i in range(n_traces):
        trace_mask = mask_gather[:, i]
        sw_percentage = (np.sum(trace_mask) / n_samples) * 100
        if sw_percentage > 0:
            print(f"  第{i:3d}道: {sw_percentage:5.2f}%")
    plt.show()
    

