import numpy as np
import matplotlib.pyplot as plt

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


# ============ 主程序 ============
# 读取原始npy数据
input_file = r'D:\桌面\面波压制\Field data shot65\npy数据\noisy.npy'  # 替换为你的输入文件名
output_file = r'D:\桌面\面波压制\Field data shot65\npy数据\mask_data.npy'  # 输出掩码文件名

# 加载数据
data = np.load(input_file)
print(f"原始数据形状: {data.shape}")
print(f"原始数据类型: {data.dtype}")

# 参数设置
dt = 0.00025  # 时间采样间隔(秒)，根据你的数据修改
window_width = 0.015  # 高斯窗口宽度
time_shift = 0.001  # 时间滑动步长
threshold = 0.05  # 能量阈值

# 创建分析器
analyzer = SeismicEnergyAnalyzer(window_width=window_width, time_shift=time_shift)

# 为每一道创建掩码
if data.ndim == 1:
    # 单道数据
    mask = analyzer.create_trace_mask(data, dt, threshold)
elif data.ndim == 2:
    # 多道数据
    n_samples, n_traces = data.shape
    mask = np.zeros_like(data, dtype=int)
    
    print(f"处理 {n_traces} 道数据...")
    for i in range(n_traces):
        mask[:, i] = analyzer.create_trace_mask(data[:, i], dt, threshold)
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{n_traces} 道")
else:
    raise ValueError(f"不支持的数据维度: {data.ndim}，应为1D或2D")

print(f"\n掩码数据形状: {mask.shape}")
print(f"掩码中1的数量: {np.sum(mask)}")
print(f"掩码中0的数量: {np.sum(mask == 0)}")
print(f"掩码覆盖率: {100 * np.sum(mask) / mask.size:.2f}%")

# 保存掩码为npy文件
np.save(output_file, mask)
print(f"\n掩码已保存到: {output_file}")

# 绘制对比图
if data.ndim == 2:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 原始数据
    im1 = axes[0, 0].imshow(data, cmap='seismic', aspect='auto')
    axes[0, 0].set_title('原始面波数据')
    axes[0, 0].set_xlabel('道数')
    axes[0, 0].set_ylabel('时间采样点')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 掩码数据
    im2 = axes[0, 1].imshow(mask, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'能量掩码 (阈值={threshold})')
    axes[0, 1].set_xlabel('道数')
    axes[0, 1].set_ylabel('时间采样点')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 应用掩码后的数据
    masked_data = data * mask
    im3 = axes[1, 0].imshow(masked_data, cmap='seismic', aspect='auto')
    axes[1, 0].set_title('掩码应用效果')
    axes[1, 0].set_xlabel('道数')
    axes[1, 0].set_ylabel('时间采样点')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 掩码叠加在原始数据上（半透明显示）
    axes[1, 1].imshow(data, cmap='seismic', aspect='auto')
    # 创建掩码的半透明叠加层，掩码区域显示为红色
    mask_overlay = np.ma.masked_where(mask == 0, mask)
    im4 = axes[1, 1].imshow(mask_overlay, cmap='Reds', alpha=0.5, aspect='auto')
    axes[1, 1].set_title('掩码位置叠加显示（红色区域）')
    axes[1, 1].set_xlabel('道数')
    axes[1, 1].set_ylabel('时间采样点')
    
    plt.tight_layout()
    # plt.savefig('mask_comparison.png', dpi=300, bbox_inches='tight')
    print("对比图已显示")
    plt.show()
    
    # 额外绘制一个大图：掩码边界叠加显示
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 显示原始数据
    im = ax.imshow(data, cmap='seismic', aspect='auto')
    plt.colorbar(im, ax=ax, label='振幅')
    
    # 找到掩码边界并绘制轮廓
    from scipy import ndimage
    # 计算掩码边界
    mask_edges = ndimage.sobel(mask.astype(float))
    mask_contour = np.abs(mask_edges) > 0
    
    # 将掩码区域叠加为半透明红色
    mask_overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(mask_overlay, cmap='Blacks', alpha=0.3, aspect='auto')
    
    # 绘制掩码边界
    contour_overlay = np.ma.masked_where(~mask_contour, mask_contour)
    ax.imshow(contour_overlay, cmap='spring', alpha=0.8, aspect='auto')
    
    ax.set_title('原始炮集 + 掩码区域叠加（红色=掩码区域，亮线=边界）')
    ax.set_xlabel('道数')
    ax.set_ylabel('时间采样点')
    
    plt.tight_layout()
    print("掩码叠加大图已显示")
    plt.show()
    
elif data.ndim == 1:
    # 单道数据的可视化
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    t = np.arange(len(data)) * dt
    
    # 原始数据
    axes[0].plot(t, data, 'b-', linewidth=0.5)
    axes[0].set_title('原始地震道')
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('振幅')
    axes[0].grid(True, alpha=0.3)
    
    # 掩码
    axes[1].fill_between(t, 0, mask, where=mask>0, alpha=0.5, color='red')
    axes[1].set_title(f'能量掩码 (阈值={threshold})')
    axes[1].set_xlabel('时间 (s)')
    axes[1].set_ylabel('掩码值')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    
    # 应用掩码后的数据
    masked_data = data * mask
    axes[2].plot(t, masked_data, 'r-', linewidth=0.5)
    axes[2].set_title('掩码应用效果')
    axes[2].set_xlabel('时间 (s)')
    axes[2].set_ylabel('振幅')
    axes[2].grid(True, alpha=0.3)
    
    # 掩码叠加在原始数据上
    axes[3].plot(t, data, 'b-', linewidth=0.5, label='原始数据')
    axes[3].fill_between(t, data.min(), data.max(), where=mask>0, 
                         alpha=0.3, color='red', label='掩码区域')
    axes[3].set_title('掩码位置叠加显示')
    axes[3].set_xlabel('时间 (s)')
    axes[3].set_ylabel('振幅')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('mask_comparison.png', dpi=300, bbox_inches='tight')
    print("对比图已显示")
    plt.show()