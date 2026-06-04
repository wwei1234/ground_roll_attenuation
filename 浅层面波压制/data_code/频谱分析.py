import segyio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def remove_direct_wave(data, center_trace=22, velocity=800, dt=0.00025, mute_time0=0.010, dx = 2, taper=20, plot=False):
    n_time, n_trace = data.shape
    muted_data = data.copy()
    mask = np.ones_like(data)

    # 计算每个道的直达波走时
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

def read_segy(data_dir,shotnum=0):
    with segyio.open(data_dir,'r',ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX))
        len_shot = trace_num//shot_num
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num,time,len_shot))
        for j in range(0,shot_num):
            data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def agc(data, window_len=30):
    agc_data = np.zeros_like(data)
    half_win = window_len // 2
    for i in range(data.shape[1]):
        trace = data[:, i]
        agc_trace = np.zeros_like(trace)
        for j in range(len(trace)):
            start = max(0, j - half_win)
            end = min(len(trace), j + half_win)
            window = trace[start:end]
            rms = np.sqrt(np.mean(window ** 2)) + 1e-8
            agc_trace[j] = trace[j] / rms
        agc_data[:, i] = agc_trace

    min_val = agc_data.min()
    max_val = agc_data.max()
    agc_data = (agc_data - min_val) / (max_val - min_val + 1e-8)
    return agc_data

def plot_spectrum(data, dt, title="频谱分析"):
    """绘制平均频谱（仅显示0–120 Hz）"""
    n_time, n_trace = data.shape
    freqs = np.fft.rfftfreq(n_time, dt)
    spectrum = np.zeros_like(freqs)

    for i in range(n_trace):
        fft_trace = np.fft.rfft(data[:, i])
        spectrum += np.abs(fft_trace)
    spectrum /= n_trace  # 平均化

    # 只取0–120 Hz范围
    freq_limit = 150
    idx = freqs <= freq_limit

    plt.figure(figsize=(8, 4))
    plt.plot(freqs[idx], spectrum[idx], color='steelblue')
    plt.title(title)
    plt.xlabel("频率 (Hz)")
    plt.ylabel("幅度")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def bandpass_filter(data, dt, lowcut, highcut, order=4):
    """Butterworth 带通滤波"""
    nyquist = 0.5 / dt
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    return filtered_data

# segyfile = r"D:\桌面\面波压制\data\f20_z_test+GathEP.sgy"
# data = read_segy(segyfile, shotnum=700)
# data = data[2]
# # 去直达波
# data = remove_direct_wave(data, center_trace=22, velocity=800, dt=0.00025, mute_time0=0.010, dx=2, taper=20, plot=False)
# 频谱分析（去直达波前后对比）

data = np.load(r"D:\桌面\面波压制\Synthetic data test 2\npy文件汇总\clean.npy")
print("绘制去直达波前后的频谱分析：")
plot_spectrum(data, dt=0.00025, title="去直达波后频谱")
# 带通滤波（例如 5–80 Hz）
# filtered_data = bandpass_filter(data, dt=0.00025, lowcut=40, highcut=60, order=4)
# # 绘制滤波后结果与频谱
# plt.figure(figsize=(10, 6))
# vmax = np.max(np.abs(filtered_data)) * 0.8
# plt.imshow(filtered_data, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
# plt.title("带通滤波后炮集 (5–80 Hz)")
# plt.xlabel("道号")
# plt.ylabel("时间采样点")
# plt.tight_layout()
# plt.show()
# print("绘制带通滤波后的频谱分析：")
# plot_spectrum(filtered_data, dt=0.00025, title="带通滤波后频谱")

