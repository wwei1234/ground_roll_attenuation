import numpy as np
import matplotlib.pyplot as plt
import segyio
from matplotlib import rcParams

# 设置字体为Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['axes.unicode_minus'] = False

class SeismicEnergyAnalyzer:
    """地震数据能量片段分析器"""
    
    def __init__(self, window_width=0.08, time_shift=0.01):
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
    
    def plot_three_figures_separate(self, seismic_trace, dt, time_centers, energies, 
                          threshold=0.001, window_index=None, interactive=False, save_figures=False, save_dir='./'):
        """
        分别绘制三张独立的图：
        1. 原始地震道 + 某一时刻的高斯时窗曲线
        2. 该高斯时窗对应的地震道片段（时间轴范围为原始道范围）
        3. 地震道平均能量统计图
        
        参数:
        seismic_trace: 地震道数据
        dt: 采样间隔
        time_centers: 时间中心
        energies: 能量值
        threshold: 阈值
        window_index: 选择哪个时窗（如果为None，自动选择能量最大的）
        interactive: 是否启用交互式选择模式
        save_figures: 是否保存图片（PNG和矢量图格式）
        save_dir: 保存目录
        
        返回:
        fig1, fig2, fig3: 三个独立的图形对象
        window_index: 选择的窗口索引
        """
        t = np.arange(len(seismic_trace)) * dt
        alpha = 1.0 / self.window_width
        
        center_time = time_centers[window_index]
        
        # 生成选定的高斯窗
        gaussian_win = self.gaussian_window(t, center_time, alpha)
        
        # 生成选定的地震道片段
        segment = seismic_trace * gaussian_win
        
        # 识别面波区域
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        
        # ========== 第一张图：原始地震道 + 高斯时窗 ==========
        fig1 = plt.figure(figsize=(12, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(t, seismic_trace, 'k-', linewidth=1.5, label='Original seismic trace')
        
        # 为了可视化，将高斯窗缩放到与地震道振幅相当的范围
        gaussian_win_scaled = gaussian_win * np.max(np.abs(seismic_trace)) / np.max(gaussian_win)
        ax1.plot(t, gaussian_win_scaled, 'k--', linewidth=1.5, 
                label=f'Gaussian window (center={center_time:.3f}s)')
        
        # 标记面波时间段
        if np.any(surface_wave_mask):
            surface_wave_times = time_centers[surface_wave_mask]
            for sw_time in surface_wave_times:
                ax1.axvspan(sw_time - self.window_width/2, 
                          sw_time + self.window_width/2,
                          alpha=0.2, color='gray')
        
        ax1.set_xlabel('Time (s)', fontsize=18)
        ax1.set_ylabel('Amplitude', fontsize=18)
        ax1.tick_params(labelsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.set_xlim([t[0], t[-1]])  # 设置x轴范围，去除空白
        plt.tight_layout()
        
        # ========== 第二张图：地震道片段 ==========
        fig2 = plt.figure(figsize=(12, 6))
        ax2 = fig2.add_subplot(111)
        ax2.plot(t, segment, 'k-', linewidth=1.5, label='Seismic trace segment')
        ax2.set_xlabel('Time (s)', fontsize=18)
        ax2.set_ylabel('Amplitude', fontsize=18)
        ax2.tick_params(labelsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12, loc='upper right')
        ax2.set_xlim([t[0], t[-1]])  # 保持与原始地震道相同的时间范围
        plt.tight_layout()
        
        # ========== 第三张图：能量统计图 ==========
        fig3 = plt.figure(figsize=(12, 6))
        ax3 = fig3.add_subplot(111)
        ax3.plot(time_centers, energies, 'k-', linewidth=2, label='Segment energy')
        ax3.axhline(y=threshold, color='k', linestyle='--', 
                   linewidth=1.5, label=f'Surface wave threshold = {threshold}')
        
        # 标记面波区域
        if np.any(surface_wave_mask):
            ax3.fill_between(time_centers, 0, energies, 
                           where=surface_wave_mask, 
                           alpha=0.3, color='gray', 
                           label='Identified surface wave region')
        
        ax3.set_xlabel('Time (s)', fontsize=18)
        ax3.set_ylabel('Average amplitude energy', fontsize=18)
        ax3.tick_params(labelsize=16)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=12, loc='upper right')
        ax3.set_xlim([time_centers[0], time_centers[-1]])  # 设置x轴范围，去除空白
        plt.tight_layout()
        
        # 保存图片
        if save_figures:
            import os
            # 确保保存目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            print(f"\n正在保存图片到目录: {save_dir}")
            
            # 保存第一张图
            fig1.savefig(os.path.join(save_dir, 'figure1_trace_and_window.png'), 
                        dpi=300, bbox_inches='tight')
            fig1.savefig(os.path.join(save_dir, 'figure1_trace_and_window.pdf'), 
                        bbox_inches='tight')
            fig1.savefig(os.path.join(save_dir, 'figure1_trace_and_window.eps'), 
                        bbox_inches='tight')
            print("  ✓ 图1保存完成: figure1_trace_and_window (PNG, PDF, SVG)")
            
            # 保存第二张图
            fig2.savefig(os.path.join(save_dir, 'figure2_segment.png'), 
                        dpi=300, bbox_inches='tight')
            fig2.savefig(os.path.join(save_dir, 'figure2_segment.pdf'), 
                        bbox_inches='tight')
            fig2.savefig(os.path.join(save_dir, 'figure2_segment.eps'), 
                        bbox_inches='tight')
            print("  ✓ 图2保存完成: figure2_segment (PNG, PDF, SVG)")
            
            # 保存第三张图
            fig3.savefig(os.path.join(save_dir, 'figure3_energy.png'), 
                        dpi=300, bbox_inches='tight')
            fig3.savefig(os.path.join(save_dir, 'figure3_energy.pdf'), 
                        bbox_inches='tight')
            fig3.savefig(os.path.join(save_dir, 'figure3_energy.eps'), 
                        bbox_inches='tight')
            print("  ✓ 图3保存完成: figure3_energy (PNG, PDF, SVG)")
            print(f"\n所有图片已保存! (PNG格式300dpi, PDF和SVG矢量格式)")
        
        return fig1, fig2, fig3, window_index
    
    def plot_energy_statistics(self, time_centers, energies, threshold=0.001, 
                              title='Seismic trace segment energy statistics'):
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
        plt.plot(time_centers, energies, 'b-', linewidth=2, label='Segment energy')
        
        # 绘制阈值线
        plt.axhline(y=threshold, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Surface wave threshold = {threshold}')
        
        # 标记面波区域
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        if np.any(surface_wave_mask):
            plt.fill_between(time_centers, 0, energies, 
                           where=surface_wave_mask, 
                           alpha=0.3, color='gray', 
                           label='Identified surface wave region')
        
        plt.xlabel('Time (s)', fontsize=18)
        plt.ylabel('Average amplitude energy', fontsize=18)
        plt.tick_params(labelsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim([time_centers[0], time_centers[-1]])  # 设置x轴范围，去除空白
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
        ax1.set_xlabel('Time (s)', fontsize=18)
        ax1.set_ylabel('Amplitude', fontsize=18)
        ax1.tick_params(labelsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([t[0], t[-1]])  # 设置x轴范围，去除空白
        
        # 标记面波时间段
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        if np.any(surface_wave_mask):
            surface_wave_times = time_centers[surface_wave_mask]
            for tw_time in surface_wave_times:
                ax1.axvspan(tw_time - self.window_width/2, 
                          tw_time + self.window_width/2,
                          alpha=0.2, color='gray')
        
        # 绘制能量统计
        ax2.plot(time_centers, energies, 'b-', linewidth=2, label='Segment energy')
        ax2.axhline(y=threshold, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Threshold = {threshold}')
        ax2.fill_between(time_centers, 0, energies, 
                        where=surface_wave_mask, 
                        alpha=0.3, color='gray', 
                        label='Surface wave region')
        ax2.set_xlabel('Time (s)', fontsize=18)
        ax2.set_ylabel('Average amplitude energy', fontsize=18)
        ax2.tick_params(labelsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.set_xlim([time_centers[0], time_centers[-1]])  # 设置x轴范围，去除空白
        
        plt.tight_layout()
        return fig
    
def read_segy(data_dir,shotnum=0):
    with segyio.open(data_dir,'r',ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX) #number of all trace
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX)) #shot number 
        len_shot = trace_num//shot_num   #The length of the data in each shot data
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num,time,len_shot))
        for j in range(0,shot_num):
            data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data


# 使用示例
if __name__ == "__main__":
    # 生成模拟地震数据
    dt = 0.00025  # 采样间隔 (秒)
    seismic_trace = read_segy(r"D:\桌面\面波压制\data\f20_z_test+GathEP.sgy", shotnum=700)[0][:, 32]
    print(seismic_trace.shape)
    seismic_trace = seismic_trace / np.max(np.abs(seismic_trace))  # 归一化处理
 
    # 创建分析器
    analyzer = SeismicEnergyAnalyzer(window_width=0.01, time_shift=0.002)
    
    # 计算能量统计
    time_centers, energies, segments = analyzer.compute_segment_energy(
        seismic_trace, dt
    )
    
    # 设置阈值并识别面波
    threshold = 0.08
    surface_wave_mask = analyzer.identify_surface_wave(energies, threshold)
    
    # 输出面波时间范围
    if np.any(surface_wave_mask):
        sw_times = time_centers[surface_wave_mask]
        print(f"识别的面波时间范围: {sw_times[0]:.2f}s ~ {sw_times[-1]:.2f}s")
    
    # 绘制三张独立的图
    # 直接在代码中指定窗口索引或时间
    
    # 方式1：直接指定窗口索引
    fig1, fig2, fig3, selected_idx = analyzer.plot_three_figures_separate(
        seismic_trace, dt, time_centers, energies, threshold, 
        window_index=50,  # 直接指定窗口索引
        save_figures=True,
        save_dir=r'D:\桌面\面波压制\矢量图2'
    )
    
    
    print(f"\n显示的窗口索引: {selected_idx}, 中心时间: {time_centers[selected_idx]:.3f}s")
    
    plt.show()
    
    print("\n程序说明:")
    print("1. plot_three_figures_separate() 方法分别绘制三张独立的图")
    print("2. 三种使用方式:")
    print("   - interactive=True: 启用交互式选择，可以输入窗口索引或时间值（如'0.5s'）")
    print("   - window_index=数字: 直接指定窗口索引")
    print("   - 不指定参数: 自动选择能量最大的窗口")
    print("3. 图形样式:")
    print("   - 字体: Times New Roman")
    print("   - 坐标轴标签字号: 18")
    print("   - 刻度字号: 16")
    print("   - 所有标签使用英文")
    print("4. 自动保存功能:")
    print("   - save_figures=True: 自动保存图片")
    print("   - save_dir='目录路径': 指定保存目录")
    print("   - 每张图保存为3种格式: PNG (300dpi), PDF (矢量), SVG (矢量)")