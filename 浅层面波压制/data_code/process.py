import numpy as np
import segyio
import os
import matplotlib.pyplot as plt

# ===================== 逐道归一化 =====================
def normalize_traces_per_trace(shot_gather):
    """逐道归一化：每道振幅除以本道最大绝对值"""
    normalized_gather = np.copy(shot_gather)
    n_samples, n_traces = normalized_gather.shape
    
    for i in range(n_traces):
        trace = normalized_gather[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized_gather[:, i] = trace / max_amp
    
    return normalized_gather


# ===================== 直达波去除 =====================
def remove_direct_wave(data, center_trace=24, velocity=1200, dt=0.00025, 
                       mute_time0=0.025, dx=2, taper=0):
    """
    去除直达波（线性走时 + taper 过渡）
    data: [time, trace]
    """
    n_time, n_trace = data.shape
    muted_data = data.copy()
    mask = np.ones_like(data)

    for tr in range(n_trace):
        offset = abs(tr - center_trace) * dx
        t0 = mute_time0 + offset / velocity
        t0_idx = int(t0 / dt)

        if t0_idx < n_time:
            mask[:t0_idx, tr] = 0

            taper_end = min(t0_idx + taper, n_time)
            for t in range(t0_idx, taper_end):
                mask[t, tr] = 1 - (t - t0_idx) / taper

    muted_data *= mask
    return muted_data


# ===================== 读取 SEGY 中的某一个炮集 =====================
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

# data = np.load(r"D:\桌面\面波压制\实际数据结果\DMSSL6\after_filter.npy")
data = read_segy(r"传统方法\wl_cs20.sgy", 1)[0]

result = remove_direct_wave(data, center_trace=24, velocity=2400, dt=0.00025, mute_time0=0.027, dx=2, taper=0)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("原始噪声炮集")
plt.imshow(data, cmap='seismic', aspect='auto', vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
plt.colorbar(label='振幅')
plt.subplot(1, 2, 2)
plt.title("去除直达波后噪声炮集")
plt.imshow(result, cmap='seismic', aspect='auto', vmin=-np.max(np.abs(result)), vmax=np.max(np.abs(result)))
plt.colorbar(label='振幅')
plt.title("去除直达波后噪声炮集（归一化）")
plt.tight_layout()
plt.show()

np.save(r"直达波切除\wl_cs_free", result)
# # ===================== 主处理流程 =====================
# def process_single_shot(input_sgy, output_npy, target_shot,
#                         center_trace=24, velocity=1200, dt=0.00025,
#                         mute_time0=0.010, dx=2, taper=20):
#     """
#     处理单炮集：
#     1. 从SEGY读取第 target_shot 炮
#     2. 直达波去除
#     3. 逐道归一化
#     4. 保存
#     """

#     # 读取炮集
#     shot = read_segy(input_sgy, target_shot)[60][0:1200,:]

#     # 直达波去除
#     shot_no_direct = remove_direct_wave(
#         shot,
#         center_trace=center_trace,
#         velocity=velocity,
#         dt=dt,
#         mute_time0=mute_time0,
#         dx=dx,
#         taper=taper
#     )

#     # 逐道归一化
#     shot_norm = normalize_traces_per_trace(shot_no_direct)

#     np.save(output_npy, shot_no_direct)

#     plt.figure(figsize=(10, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("去除直达波后炮集")
#     plt.imshow(shot_no_direct, cmap='seismic', aspect='auto', vmin=-np.max(np.abs(shot_no_direct)), vmax=np.max(np.abs(shot_no_direct)))
#     plt.colorbar(label='振幅')
#     plt.subplot(1, 2, 2)
#     plt.title("逐道归一化后炮集")
#     plt.imshow(shot_norm, cmap='seismic', aspect='auto', vmin=-1, vmax=1)
#     plt.colorbar(label='归一化振幅')
#     plt.tight_layout()
#     plt.show()

#     # np.save(output_npy, shot_norm)
#     print(f"处理完毕！已将第 {target_shot} 炮保存为：{output_npy}")

 
# # ========== 使用示例 ==========
# # 读取并处理第 0 炮
# process_single_shot(r"D:\桌面\面波压制\训练数据\FK_sn3_1.sgy", r"D:\桌面\面波压制\实际数据结果\F-K\F-K_result.npy", target_shot=168)

