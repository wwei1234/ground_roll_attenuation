import numpy as np
import segyio
import matplotlib.pyplot as plt
import scipy.signal as sps
import scipy.ndimage as ndi
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def read_segy(data_dir,shotnum=5):
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

    # 归一化到 [0, 1]
    min_val = agc_data.min()
    max_val = agc_data.max()
    agc_data = (agc_data - min_val) / (max_val - min_val + 1e-8)
    return agc_data

def remove_direct_wave(data, center_trace=64.5, velocity=0.18, dt=0.001, mute_time0=10, taper=20, plot=True):
    n_time, n_trace = data.shape
    muted_data = data.copy()
    mask = np.ones_like(data)

    # 计算每个道的直达波走时
    for tr in range(n_trace):
        offset = abs(tr - center_trace)  # 支持小数中心
        t0 = mute_time0 + offset / velocity
        t0_idx = int(t0)

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

def remove_below_surface_wave(data, t0=80, cut_slope=0.5, plot=True):
    """
    去除面波下方能量，保留纯面波部分。
    
    参数：
        data: np.ndarray, shape=(time, trace)，输入炮集记录
        t0: float，炮点附近（中心道）开始的切除起点时间（单位：采样点）
        cut_slope: float，控制界线随偏移距的斜率（每道增加多少采样点）
        plot: bool，是否绘图显示结果

    返回：
        data_cut: np.ndarray, 去除面波下方后的数据
    """
    time, ntraces = data.shape
    data_cut = np.copy(data)
    
    # 炮点在64和65之间，中点索引为64（Python从0开始）
    center = 64
    
    # 为每个道计算切除界线
    for i in range(ntraces):
        offset = abs(i - center)
        t_cut = int(t0 + cut_slope * offset)  # 随偏移距变化的界线
        t_cut = np.clip(t_cut, 0, time)       # 防止越界
        data_cut[t_cut:, i] = 0               # 界线下部分置零
    
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(data, cmap='gray', aspect='auto')
        axs[0].set_title('原始炮集')
        axs[1].imshow(data_cut, cmap='gray', aspect='auto')
        axs[1].set_title('去除面波下方能量后')
        plt.show()
    
    return data_cut

def stft_gabor_like(trace, fs, nperseg=128, noverlap=96, window_sigma=None, nfft=None):
    """
    基于 STFT 的近似 Gabor（使用高斯或 hann 窗）。
    trace: 1D array 时间序列
    fs: 采样频率（Hz）
    nperseg: 窗长（样点）
    noverlap: 重叠（样点）
    window_sigma: 若指定则使用高斯窗；否则使用hann窗
    nfft: FFT 点数
    返回 (f, t, Sxx) 其中 Sxx 为幅值谱（线性幅值）
    """
    if nfft is None:
        nfft = max(256, nperseg)
    if window_sigma is None:
        window = 'hann'
        f, t, Zxx = sps.stft(trace, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=False, boundary=None)
        S = np.abs(Zxx)
    else:
        # 构造高斯窗
        n = np.arange(nperseg)
        center = (nperseg - 1) / 2.0
        sigma = window_sigma
        gauss = np.exp(-0.5 * ((n - center) / sigma) ** 2)
        f, t, Zxx = sps.stft(trace, fs=fs, window=gauss, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=False, boundary=None)
        S = np.abs(Zxx)
    return f, t, S

def detect_ground_roll_mask(gather,
                            dt=0.004,
                            f_low=2.0,
                            f_high=25.0,
                            nperseg=128,
                            noverlap=96,
                            window_sigma=None,
                            nfft=None,
                            energy_mode='sum',   # 'sum' 或 'max'
                            threshold_type='percentile', # 'percentile' 或 'median_std'
                            percentile=90,
                            median_std_k=1.0,
                            min_duration_samples=8,
                            morph_close_size=(5,3),
                            expand_time_samples=4):

    ntime, ntr = gather.shape
    fs = 1.0 / dt

    # 结果初始化
    mask = np.zeros_like(gather, dtype=bool)

    # 逐道处理
    energy_envelopes = np.zeros((ntime, ntr), dtype=float)

    for j in range(ntr):
        trace = gather[:, j]
        f, t_stft, S = stft_gabor_like(trace, fs, nperseg=nperseg, noverlap=noverlap, window_sigma=window_sigma, nfft=nfft)

        # 选取频带索引
        freq_idx = np.where((f >= f_low) & (f <= f_high))[0]
        if freq_idx.size == 0:
            raise ValueError("选取的频率带在STFT结果中无交集，请调整 f_low/f_high 或 nfft/nperseg。")

        # 在目标频带上聚合能量 -> 得到随时间变化的能量包络（对应 STFT 的 t_stft 点）
        if energy_mode == 'sum':
            energy_tf = S[freq_idx, :].sum(axis=0)
        else:
            energy_tf = S[freq_idx, :].max(axis=0)

        # 将 STFT 时刻映射回原始采样时间索引
        # sps.stft 的 t_stft 单位是秒，这里映射每个原始时间样点到最近的 stft 时间索引
        # 先把 energy_tf 插值到原始时间轴
        t_stft_samples = (t_stft * fs).astype(int)  # STFT 时间对应的样点索引
        # 有时候 t_stft 的最后一个点可能小于 ntime-1，使用 np.interp 插值到全部样点
        time_axis = np.arange(ntime)
        energy_interp = np.interp(time_axis, t_stft_samples, energy_tf, left=0, right=0)

        energy_envelopes[:, j] = energy_interp

    # 全局或逐道阈值决定掩码（两种策略均提供）
    mask_raw = np.zeros_like(energy_envelopes, dtype=bool)

    if threshold_type == 'percentile':
        # 使用全局百分位阈值（可以更稳健）
        thr = np.percentile(energy_envelopes, percentile)
        mask_raw = energy_envelopes >= thr
    else:
        # 对每道独立阈值：中位数 + k*std
        for j in range(ntr):
            col = energy_envelopes[:, j]
            med = np.median(col)
            std = np.std(col)
            thr = med + median_std_k * std
            mask_raw[:, j] = col >= thr

    # 去短脉冲（时间方向上最小长度过滤）
    for j in range(ntr):
        col = mask_raw[:, j].astype(int)
        # 连通域去短
        labeled, num = ndi.label(col)
        for lab in range(1, num + 1):
            inds = np.where(labeled == lab)[0]
            if inds.size < min_duration_samples:
                col[inds] = 0
        mask_raw[:, j] = col.astype(bool)

    # 在时间方向上做小的膨胀，避免边界太窄
    if expand_time_samples > 0:
        structure = np.ones((expand_time_samples * 2 + 1,))
        for j in range(ntr):
            mask_raw[:, j] = ndi.binary_dilation(mask_raw[:, j], structure=structure)

    # 合并为 mask（并做形态学闭运算消除小孔洞并沿道方向连通）
    mask_closed = ndi.binary_closing(mask_raw, structure=np.ones(morph_close_size))

    # 进一步做二维中值滤波 / 平滑，增强连通性
    mask_smooth = ndi.median_filter(mask_closed.astype(int), size=(5, 3)).astype(bool)

    return mask_smooth.astype(bool), energy_envelopes

if __name__ == "__main__":
    ntime = 1200
    ntr = 48
    # 下面只是 demo，实际请加载你的数据
    data = read_segy(r'D:\桌面\面波压制\data\dataset4_700.sgy', shotnum=700)[0][0:1200]
    # （可选）在中间人为叠加一个低频高振幅的"面波"以测试
    # t = np.arange(ntime)
    # for j in range(40, 90):
    #     data[:, j] += 2.0 * np.exp(-((t - (100 + j // 2)) / 30.0) ** 2) * np.sin(2 * np.pi * 6.0 * t * 0.002)

    # 调用检测函数（参数可按需调整）
    mask, energy = detect_ground_roll_mask(data,
                                           dt=0.0025,
                                           f_low=2.0,
                                           f_high=15.0,
                                           nperseg=128,
                                           noverlap=112,
                                           window_sigma=None,    # 或 nperseg/6 使用高斯窗
                                           nfft=256,
                                           energy_mode='sum',
                                           threshold_type='percentile', #
                                           percentile=92,  # 面波阈值百分位
                                           min_duration_samples=6,
                                           morph_close_size=(7,5),
                                           expand_time_samples=4)

    # 可视化结果（展示能量 & 掩码 & 原始炮集）
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    im0 = axes[0].imshow(data, aspect='auto', cmap='seismic', vmin=-1.0, vmax=1.0)
    axes[0].set_title("原始炮集 (ntime x ntr)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.log1p(energy), aspect='auto', cmap='viridis')
    axes[1].set_title("低频能量包络 (log1p)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(mask.astype(int), aspect='auto', cmap='gray')
    axes[2].set_title("检测到的面波掩码 (1=面波)")
    plt.colorbar(im2, ax=axes[2])
    plt.show()




