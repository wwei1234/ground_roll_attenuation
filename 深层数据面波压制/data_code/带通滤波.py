import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 读取npy数据
data = np.load(r'原始炮集\real_data_gather000.npy')  # 替换为你的文件路径
# data = np.load(r'inference_output\real_data_gather089_denoised.npy')  # 替换为你的文件路径
# save_path = r"D:\桌面\面波压制\Field data shot65\npy数据\DMSSL_result_after_filter.npy"

# 设置滤波参数
sample_rate = 250  # 采样率 (Hz)
low_freq = 20       # 低频截止 (Hz)
high_freq = 80      # 高频截止 (Hz)
filter_order = 5    # 滤波器阶数

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
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    
    # 设计巴特沃斯带通滤波器
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    # 对每一道进行滤波
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = signal.sosfiltfilt(sos, data[:, i])
    
    return filtered_data

# 应用带通滤波
filtered_data = bandpass_filter(data, low_freq, high_freq, sample_rate, filter_order)

# 绘制结果对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 原始数据
im1 = axes[0].imshow(data, cmap='seismic', aspect='auto', 
                     interpolation='bilinear', vmin=-np.percentile(np.abs(data), 95),
                     vmax=np.percentile(np.abs(data), 95))
axes[0].set_title('原始炮集记录', fontsize=14, fontweight='bold')
axes[0].set_xlabel('道号', fontsize=12)
axes[0].set_ylabel('时间采样点', fontsize=12)
plt.colorbar(im1, ax=axes[0], label='振幅')

# 滤波后数据
im2 = axes[1].imshow(filtered_data, cmap='seismic', aspect='auto',
                     interpolation='bilinear', vmin=-np.percentile(np.abs(filtered_data), 95),
                     vmax=np.percentile(np.abs(filtered_data), 95))
axes[1].set_title(f'带通滤波后 ({low_freq}-{high_freq} Hz)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('道号', fontsize=12)
axes[1].set_ylabel('时间采样点', fontsize=12)
plt.colorbar(im2, ax=axes[1], label='振幅')

plt.tight_layout()
# plt.savefig('bandpass_filter_result.png', dpi=300, bbox_inches='tight')
plt.show()

# 可选：绘制单道对比
trace_num = data.shape[1] // 2  # 选择中间一道
time = np.arange(data.shape[0]) / sample_rate  # 时间轴

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, data[:, trace_num], 'b-', linewidth=0.5)
plt.title(f'原始数据 - 第{trace_num}道', fontsize=12, fontweight='bold')
plt.ylabel('振幅', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(time, filtered_data[:, trace_num], 'r-', linewidth=0.5)
plt.title(f'滤波后数据 - 第{trace_num}道', fontsize=12, fontweight='bold')
plt.xlabel('时间 (s)', fontsize=10)
plt.ylabel('振幅', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('trace_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存滤波后的数据
# np.save(save_path, filtered_data)
print(f"滤波完成！")
print(f"数据形状: {data.shape}")
print(f"滤波参数: {low_freq}-{high_freq} Hz")
print(f"结果已保存为: filtered_shot_gather.npy")