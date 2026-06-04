import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False

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
    
def normalize_traces_global(shot_gather):
    """
    对炮集数据进行全局归一化处理，使整个炮集的最大绝对振幅为 1。

    参数:
        shot_gather (ndarray): 形状为 [n_samples, n_traces] 的二维数组。
        
    返回:
        normalized_gather (ndarray): 全局归一化后的炮集数据。
    """
    # 复制输入，避免修改原数据
    normalized_gather = np.copy(shot_gather)

    # 计算全局最大绝对振幅
    max_amp = np.max(np.abs(normalized_gather))

    # 避免除零错误
    if max_amp > 0:
        normalized_gather = normalized_gather / max_amp

    return normalized_gather

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

def remove_direct_wave(data, center_trace=22, velocity=800, dt=0.00025, mute_time0=0.010, dx = 2, taper=20, plot=False):
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

    # 归一化到 [0, 1]
    min_val = agc_data.min()      
    max_val = agc_data.max()
    return (agc_data - min_val) / (max_val - min_val + 1e-8)
                                               
def scale_to_neg1_1(A, B):
    """
    同时缩放 A 和 B，使得 A+B 的范围恰好落在 (-1,1)
    """
    C = A + B

    # 找到需要缩放的最大绝对值
    C_min = C.min()
    C_max = C.max()
    max_abs = max(abs(C_min), abs(C_max))

    # 缩放因子 k
    k = 1.0 / max_abs

    # 缩放 A 和 B
    A_scaled = k * A
    B_scaled = k * B

    # 缩放后的相加结果
    C_scaled = A_scaled + B_scaled

    return A_scaled, B_scaled, C_scaled, k
 
dt = 4  # 采样间隔(ms)

data = np.load(r"最终数据\original.npy")[0:750]


# data = data1 - data2
# data = normalize_traces_global(data)
# np.save(r"最终数据\U-Net_res.npy", data)
# data = agc(data, window_len=300)

save_path1 = r"最终图件\传统方法_result"
os.makedirs(os.path.dirname(save_path1), exist_ok=True)
# ====================== Figure 1 ======================
n_samples, n_traces = data.shape
print(f"数据维度: {n_samples} 个采样点, {n_traces} 道")
time_axis = np.arange(n_samples) * dt 
plt.figure(figsize=(3, 4))
plt.imshow(
    data,interpolation='bicubic',
    cmap='seismic',
    aspect='auto',
    extent=[0, n_traces, time_axis[-1], time_axis[0]]
) 
plt.clim(-1, 1)
# plt.colorbar()
plt.xlabel("Trace", fontsize=12)
plt.ylabel("Time / ms", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(save_path1+".png",format = "png", dpi=600)
plt.savefig(save_path1+".eps",format = 'eps', dpi=600)
plt.savefig(save_path1+".pdf",format = 'pdf', dpi=600)
plt.show()



