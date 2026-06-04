import numpy as np
import matplotlib.pyplot as plt
import segyio


def read_segy(data_dir: str, shotnum: int = 0) -> np.ndarray:
    """
    读取sgy文件中的所有地震数据，并按炮点组织成三维数组。
    注意：这里不进行归一化，保留原始数据
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
        data = np.zeros((shot_num, time, len_shot), dtype=np.float32)
        
        for j in range(0, shot_num):
            shot_data = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
            data[j, :, :] = shot_data
        
        return data

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

def trace_minmax_normalize(shot: np.ndarray) -> np.ndarray:
    """
    道级 Min-Max 归一化到 [-1,1]
    shot: (n_samples, n_traces)
    """
    out = shot.copy()
    n_samples, n_traces = shot.shape
    for i in range(n_traces):
        trace = shot[:, i]
        max_val = np.max(np.abs(trace))
        if max_val > 0:
            out[:, i] = trace / max_val
    return out

def trace_rms_normalize(shot: np.ndarray) -> np.ndarray:
    """
    道级 RMS 归一化
    shot: (n_samples, n_traces)
    """
    out = shot.copy()
    n_samples, n_traces = shot.shape
    for i in range(n_traces):
        trace = shot[:, i]
        rms = np.sqrt(np.mean(trace**2))
        if rms > 0:
            out[:, i] = trace / rms
    return out


import numpy as np

def agc_normalize(shot: np.ndarray, window_len: int = 30) -> np.ndarray:
    """
    对炮集进行道级 RMS 自动增益控制（AGC）+ [-1,1] 归一化。
    
    参数：
        shot: np.ndarray, shape=(n_samples, n_traces)
            原始炮集数据
        window_len: int
            局部 RMS 窗口长度（采样点数）

    返回：
        np.ndarray, shape=(n_samples, n_traces)
            AGC 处理后的炮集，幅值归一化到 [-1,1]
    """
    n_samples, n_traces = shot.shape
    agc_data = np.zeros_like(shot, dtype=np.float32)
    
    # 卷积核，用于滑动均方
    kernel = np.ones(window_len) / window_len

    for i in range(n_traces):
        trace = shot[:, i].astype(np.float32)
        # 局部 RMS
        local_rms = np.sqrt(np.convolve(trace**2, kernel, mode='same') + 1e-8)
        # 道级 AGC
        agc_trace = trace / local_rms
        agc_data[:, i] = agc_trace

    # 道级归一化到 [-1,1]
    max_abs = np.max(np.abs(agc_data))
    if max_abs > 0:
        agc_data = agc_data / max_abs

    return agc_data

data = read_segy(r"D:\桌面\面波压制\合成记录\data\clean\clean_001.sgy", shotnum=5)[0]#[10:, :]

# for i in range(20):
#     data[i, :] = np.zeros(data.shape[1])


# data = trace_minmax_normalize(data)

data = agc_normalize(data, window_len=200)

# data = agc(data, window_len=200)
# data = trace_rms_normalize(data)
print(data.shape)
plt.figure()    
plt.imshow(data, "seismic", aspect='auto')
plt.colorbar()
plt.show()
