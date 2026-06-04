# import segyio
# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# def remove_direct_wave(data, center_trace=22, velocity=800, dt=0.00025, mute_time0=0.010, dx = 2, taper=20, plot=False):
#     n_time, n_trace = data.shape
#     muted_data = data.copy()
#     mask = np.ones_like(data)

#     # 计算每个道的直达波走时
#     for tr in range(n_trace):
#         offset = abs(tr - center_trace)*dx  # 支持小数中心
#         t0 = mute_time0 + offset / velocity
#         t0_idx = int(t0/dt)

#         if t0_idx < n_time:
#             mask[:t0_idx, tr] = 0
#             taper_end = min(t0_idx + taper, n_time)
#             for t in range(t0_idx, taper_end):
#                 mask[t, tr] = 1 - (t - t0_idx) / taper
#     muted_data *= mask

#     # 可视化结果
#     if plot:
#         fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
#         vmax = np.max(np.abs(data)) * 0.8
#         axs[0].imshow(data, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
#         axs[0].set_title('原始炮集')
#         axs[1].imshow(muted_data, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
#         axs[1].set_title('去除直达波后')
#         for ax in axs:
#             ax.set_xlabel('道号')
#             ax.set_ylabel('时间采样点')
#             ax.axvline(center_trace, color='lime', linestyle='--', lw=1.5, label='炮点')
#         plt.tight_layout()
#         plt.show()
#     return muted_data

# def read_segy(data_dir,shotnum=0):
#     with segyio.open(data_dir,'r',ignore_geometry=True) as f:
#         sourceX = f.attributes(segyio.TraceField.SourceX)[:]
#         trace_num = len(sourceX) #number of all trace
#         if shotnum:
#             shot_num = shotnum 
#         else:
#             shot_num = len(set(sourceX)) #shot number 
#         len_shot = trace_num//shot_num   #The length of the data in each shot data
#         time = f.trace[0].shape[0]
#         print('start read segy data')
#         data = np.zeros((shot_num,time,len_shot))
#         for j in range(0,shot_num):
#             data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
#         return data

# def seismic_to_3d_array(segyfile):
#     with segyio.open(segyfile, "r", 9, 25) as sgydata:

#         print('start read segy data')
#         # 获取行列数和样本数
#         ilines = sgydata.ilines
#         xlines = sgydata.xlines
#         samples = sgydata.samples

#         # 创建一个三维 NumPy 数组
#         # 形状为 (ilines数量, xlines数量, 样本数量)
#         seismic_data = np.zeros((len(ilines), len(xlines), len(samples)))

#         # 填充数据
#         for i, iline in enumerate(ilines):
#             for j, xline in enumerate(xlines):
#                 # 读取每个道的样本数据
#                 seismic_data[i, j, :] = sgydata.iline[iline][j]  # 这里假设是以道数索引方式访问
#     return seismic_data

# def normalize_traces_per_trace(shot_gather):
#     """逐道归一化"""
#     normalized_gather = np.copy(shot_gather)
#     n_samples, n_traces = normalized_gather.shape
    
#     for i in range(n_traces):
#         trace = normalized_gather[:, i]
#         max_amp = np.max(np.abs(trace))
#         if max_amp > 0:
#             normalized_gather[:, i] = trace / max_amp
    
#     return normalized_gather

# def agc(data, window_len=30):
#     agc_data = np.zeros_like(data)
#     half_win = window_len // 2
#     for i in range(data.shape[1]):
#         trace = data[:, i]
#         agc_trace = np.zeros_like(trace)
#         for j in range(len(trace)):
#             start = max(0, j - half_win)
#             end = min(len(trace), j + half_win)
#             window = trace[start:end]
#             rms = np.sqrt(np.mean(window ** 2)) + 1e-8
#             agc_trace[j] = trace[j] / rms
#         agc_data[:, i] = agc_trace

#     # 归一化到 [0, 1]
#     min_val = agc_data.min()
#     max_val = agc_data.max()
#     agc_data = (agc_data - min_val) / (max_val - min_val + 1e-8)

#     return agc_data



# data = seismic_to_3d_array(r"D:\桌面\深层数据面波压制\data\swath_1_geometry.sgy")
# print(data.shape)
# # data = data[:, 280:372]
# # data = remove_direct_wave(data, center_trace=24, velocity=1200, dt=0.00025, mute_time0=0.027, dx = 2, taper=0, plot=True)
# # data = data[100:600,:]
# # np.save(r"D:\桌面\面波压制\Synthetic data test2\data\noise", data)
# # data = normalize_traces_per_trace(data)
# # plt.figure()
# # plt.imshow(data, "seismic", aspect='auto')
# # plt.colorbar()
# # # plt.savefig(r"D:\桌面\Denoise\可视化\合成地震记录", dpi =2400)
# # plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         


import segyio
import numpy as np
import matplotlib.pyplot as plt

def normalize_traces_per_trace(shot_gather):
    """逐道归一化"""
    normalized_gather = np.copy(shot_gather)
    n_samples, n_traces = normalized_gather.shape
    
    for i in range(n_traces):
        trace = normalized_gather[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized_gather[:, i] = trace / max_amp
    
    return normalized_gather

def read_prestack_sgy_by_gather(segyfile):
    """
    按炮集（gather）读取预栈SGY文件，返回列表，每个元素是一个炮集
    """
    with segyio.open(segyfile, ignore_geometry=True) as f:
        nsamples = len(f.samples)          # 样本数，如3000
        sample_interval = f.samples[1] - f.samples[0]  # 采样间隔，单位ms，通常2ms
        
        # 预先提取所有道的FFID，方便分组
        ffids = np.array([h[9] for h in f.header])   # 字节9开始的FFID
        unique_ffids = np.unique(ffids)
        
        gathers = []  # 存储每个炮集
        
        for ffid in unique_ffids:
            # 找到属于当前FFID的所有道索引
            trace_indices = np.where(ffids == ffid)[0]
            ntraces = len(trace_indices)
            
            # 初始化当前炮集数据 (ntraces × nsamples)
            gather_data = np.zeros((ntraces, nsamples), dtype=np.float32)
            
            # 存储该炮集的道头信息（可选，方便后续使用坐标、偏移距等）
            gather_headers = []
            
            for i, trace_idx in enumerate(trace_indices):
                gather_data[i, :] = f.trace[trace_idx]
                gather_headers.append(f.header[trace_idx])
            
            # 可选：计算每道的偏移距（source - receiver距离）
            offsets = []
            for h in gather_headers:
                sx = h[73]  # Source X (Northing, 字节73-76)
                sy = h[77]  # Source Y (Easting)
                rx = h[81]  # Receiver X
                ry = h[85]  # Receiver Y
                # 坐标缩放（scaler）
                coord_scaler = h[71] if h[71] > 0 else 1.0 / abs(h[71]) if h[71] < 0 else 1.0
                offset = np.sqrt((sx - rx)**2 + (sy - ry)**2) * coord_scaler
                offsets.append(offset)
            offsets = np.array(offsets)
            
            gathers.append({
                'ffid': ffid,
                'data': gather_data,          # shape: (ntraces, nsamples)
                'headers': gather_headers,
                'offsets': offsets,           # 偏移距数组
                'ntraces': ntraces
            })
            
            print(f"FFID {ffid}: {ntraces} 道")
        
        print(f"总共读取 {len(gathers)} 个炮集")
        return gathers, sample_interval

# 使用示例
segy_path = r"D:\桌面\深层数据面波压制\data\001_rawdata.sgy"
gathers, dt_ms = read_prestack_sgy_by_gather(segy_path)

# 查看第10个炮集并绘图
gather10 = gathers[1]  # 第10个炮（索引从0开始）
data = gather10['data'].T
offsets = gather10['offsets']
data = data[:, 0:119]
print(data.shape)


# np.save(r"D:\桌面\深层数据面波压制\npy\real_data_gather182", data)
# data = normalize_traces_per_trace(data)
print(data.shape)
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='seismic', aspect='auto')
plt.colorbar(label='Amplitude')
plt.xlabel('Offset (feet)')
plt.ylabel('Time (s)')
plt.title(f"Shot Gather FFID {gather10['ffid']} - {gather10['ntraces']} traces")
plt.show()