# import numpy as np
# import segyio
# import os
# import matplotlib.pyplot as plt

# # ===================== 逐道归一化 =====================
# def normalize_traces_per_trace(shot_gather):
#     """逐道归一化：每道振幅除以本道最大绝对值"""
#     normalized_gather = np.copy(shot_gather)
#     n_samples, n_traces = normalized_gather.shape
    
#     for i in range(n_traces):
#         trace = normalized_gather[:, i]
#         max_amp = np.max(np.abs(trace))
#         if max_amp > 0:
#             normalized_gather[:, i] = trace / max_amp
    
#     return normalized_gather


# # ===================== 直达波去除 =====================
# def remove_direct_wave(data, center_trace=24, velocity=1200, dt=0.00025, 
#                        mute_time0=0.025, dx=2, taper=0):
#     """
#     去除直达波（线性走时 + taper 过渡）
#     data: [time, trace]
#     """
#     n_time, n_trace = data.shape
#     muted_data = data.copy()
#     mask = np.ones_like(data)

#     for tr in range(n_trace):
#         offset = abs(tr - center_trace) * dx
#         t0 = mute_time0 + offset / velocity
#         t0_idx = int(t0 / dt)

#         if t0_idx < n_time:
#             mask[:t0_idx, tr] = 0

#             taper_end = min(t0_idx + taper, n_time)
#             for t in range(t0_idx, taper_end):
#                 mask[t, tr] = 1 - (t - t0_idx) / taper

#     muted_data *= mask
#     return muted_data


# # ===================== 读取 SEGY 中的某一个炮集 =====================
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

# # data = read_segy(r"原始炮集\real_data_gather139.npy", shotnum=1)[0]
# data = np.load(r"最终数据\original.npy")[0:750]
# result = remove_direct_wave(data, center_trace=83, velocity=1600, dt=0.00025, mute_time0=0.012, dx=2, taper=0)

# plt.figure(figsize=(12, 8))
# plt.subplot(1, 2, 1)
# plt.title("原始噪声炮集")
# plt.imshow(data, cmap='seismic', aspect='auto', vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
# plt.colorbar(label='振幅')
# plt.subplot(1, 2, 2)
# plt.title("去除直达波后噪声炮集")
# plt.imshow(result, cmap='seismic', aspect='auto', vmin=-np.max(np.abs(result)), vmax=np.max(np.abs(result)))
# plt.colorbar(label='振幅')
# plt.title("去除直达波后噪声炮集（归一化）")
# plt.tight_layout()
# plt.show()
# # np.save(r"最终数据\original_without_norm.npy", data)


import numpy as np
import segyio
import os
import matplotlib.pyplot as plt

# ===================== 逐道归一化 =====================
def normalize_traces_per_trace(shot_gather):
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


# ===================== 批量处理 =====================
def batch_remove_direct_wave(input_dir, output_dir,
                              time_slice=750,
                              center_trace=83, velocity=1600,
                              dt=0.00025, mute_time0=0.012,
                              dx=2, taper=0,
                              visualize=False):
    """
    批量对文件夹中的 .npy 文件进行直达波切除
    
    input_dir:   输入文件夹路径
    output_dir:  输出文件夹路径
    time_slice:  时间轴截取长度（0表示不截取）
    visualize:   是否保存对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"共找到 {len(files)} 个 .npy 文件，开始处理...")

    for idx, fname in enumerate(files):
        fpath = os.path.join(input_dir, fname)
        data = np.load(fpath)

        # 兼容 (time, trace) 和 (shot, time, trace) 两种格式
        if data.ndim == 3:
            data = data[0]

        # 时间轴截取
        if time_slice and time_slice < data.shape[0]:
            data = data[0:time_slice]

        result = remove_direct_wave(data,
                                    center_trace=center_trace,
                                    velocity=velocity,
                                    dt=dt,
                                    mute_time0=mute_time0,
                                    dx=dx,
                                    taper=taper)

        # 保存结果
        out_fname = os.path.splitext(fname)[0] + '_muted.npy'
        np.save(os.path.join(output_dir, out_fname), result)

        # 可选：保存对比图
        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))
            vmax = np.max(np.abs(data))
            axes[0].imshow(data, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
            axes[0].set_title(f"原始炮集 - {fname}")
            axes[0].set_xlabel("道号")
            axes[0].set_ylabel("时间采样点")
            vmax2 = np.max(np.abs(result))
            axes[1].imshow(result, cmap='seismic', aspect='auto', vmin=-vmax2, vmax=vmax2)
            axes[1].set_title("直达波切除后")
            axes[1].set_xlabel("道号")
            plt.tight_layout()
            fig_path = os.path.join(output_dir, 'figures', os.path.splitext(fname)[0] + '.png')
            plt.savefig(fig_path, dpi=100)
            plt.close()

        print(f"[{idx+1}/{len(files)}] {fname} -> {out_fname}")

    print("批量处理完成。")


# ===================== 主程序 =====================
if __name__ == "__main__":
    batch_remove_direct_wave(
        input_dir=r"最终数据",        # 输入文件夹
        output_dir=r"最终数据_直达波切除",       # 输出文件夹
        time_slice=750,                      # 时间轴截取长度，0表示不截取
        center_trace=83,
        velocity=1400,
        dt=0.00025,
        mute_time0=0.016,
        dx=2,
        taper=0,
        visualize=False                      # 改为True可保存每个炮集的对比图
    )