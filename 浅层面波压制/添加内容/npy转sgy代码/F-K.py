# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal

# def fk_spectrum(data, dt, dx, window=True, freq_range=None, k_range=None, save_path=None, 
#                 normalize=True, n_samples=None):
#     """
#     计算并绘制F-K谱
    
#     参数:
#     data: 2D数组，形状为(nt, nx)，nt为时间采样点数，nx为道数
#     dt: 时间采样间隔(秒)
#     dx: 道间距(米)
#     window: 是否应用窗函数
#     freq_range: 频率显示范围，格式为[fmin, fmax]，None表示显示全部正频率
#     k_range: 波数显示范围，格式为[kmin, kmax]，None表示显示全部波数
#     save_path: 保存F-K谱数据的路径，None表示不保存
#     normalize: 是否进行逐道归一化
#     n_samples: 截取的采样点数，None表示不截取
#     """
#     # 数据预处理：逐道归一化
#     if normalize:
#         print("正在进行逐道归一化...")
#         data_processed = np.zeros_like(data)
#         for i in range(data.shape[1]):  # 遍历每一道
#             trace = data[:, i]
#             trace_max = np.max(np.abs(trace))
#             if trace_max > 0:  # 避免除以0
#                 data_processed[:, i] = trace / trace_max
#             else:
#                 data_processed[:, i] = trace
#         data = data_processed
#         print("逐道归一化完成")
    
#     # 数据预处理：截取前n个采样点
#     if n_samples is not None:
#         if n_samples < data.shape[0]:
#             print(f"截取前 {n_samples} 个采样点...")
#             data = data[:n_samples, :]
#             print(f"截取后数据形状: {data.shape}")
#         else:
#             print(f"警告: 指定的采样点数 {n_samples} 大于等于数据长度 {data.shape[0]}，不进行截取")
    
#     nt, nx = data.shape
    
#     # 应用窗函数减少频谱泄漏
#     if window:
#         # 时间方向Hanning窗
#         win_t = np.hanning(nt).reshape(-1, 1)
#         # 空间方向Hanning窗
#         win_x = np.hanning(nx).reshape(1, -1)
#         data_windowed = data * win_t * win_x
#     else:
#         data_windowed = data
    
#     # 2D FFT
#     fk = np.fft.fft2(data_windowed)
#     fk = np.fft.fftshift(fk)  # 移到中心
    
#     # 计算幅度谱(对数刻度以便显示)
#     fk_amp = np.abs(fk)
#     fk_amp_db = 20 * np.log10(fk_amp + 1e-10)  # 加小值避免log(0)
    
#     # 频率和波数轴
#     freqs = np.fft.fftfreq(nt, dt)
#     freqs = np.fft.fftshift(freqs)
    
#     kx = np.fft.fftfreq(nx, dx)
#     kx = np.fft.fftshift(kx)
    
#     # 绘图
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     # 左图：原始炮集记录
#     extent_data = [0, nx*dx, nt*dt, 0]
#     im1 = ax1.imshow(data, aspect='auto', cmap='seismic', 
#                      extent=extent_data, vmin=-np.percentile(np.abs(data), 99),
#                      vmax=np.percentile(np.abs(data), 99))
#     ax1.set_xlabel('偏移距 (m)')
#     ax1.set_ylabel('时间 (s)')
#     ax1.set_title('原始炮集记录')
#     plt.colorbar(im1, ax=ax1, label='振幅')
    
#     # 右图：F-K谱
#     extent_fk = [kx.min(), kx.max(), freqs.max(), freqs.min()]  # 交换频率范围使大频率在下
#     im2 = ax2.imshow(fk_amp_db, aspect='auto', cmap='jet',
#                      extent=extent_fk, origin='upper')  # 改为upper
#     ax2.set_xlabel('波数 (1/m)')
#     ax2.set_ylabel('频率 (Hz)')
#     ax2.set_title('F-K谱')
#     ax2.axhline(y=0, color='w', linestyle='--', linewidth=0.5)
#     ax2.axvline(x=0, color='w', linestyle='--', linewidth=0.5)
    
#     # 设置频率显示范围
#     if freq_range is not None:
#         fmin, fmax = freq_range
#         ax2.set_ylim([fmax, fmin])  # 大频率在下，所以是[fmax, fmin]
#     else:
#         # 默认只显示正频率部分，大频率在下
#         ax2.set_ylim([freqs.max(), 0])
    
#     # 设置波数显示范围
#     if k_range is not None:
#         kmin, kmax = k_range
#         ax2.set_xlim([kmin, kmax])
#     else:
#         # 默认显示全部波数范围
#         ax2.set_xlim([kx.min(), kx.max()])
    
#     plt.colorbar(im2, ax=ax2, label='幅度 (dB)')
#     plt.tight_layout()
#     plt.show()
    
#     # 保存F-K谱数据
#     if save_path is not None:
#         fk_data = {
#             'fk_amp_db': fk_amp_db,
#             'freqs': freqs,
#             'kx': kx,
#             'dt': dt,
#             'dx': dx,
#             'data_shape': data.shape
#         }
#         np.save(save_path, fk_data)
#         print(f"F-K谱数据已保存到: {save_path}")
    
#     return fk, freqs, kx


# # 示例：生成合成数据
# def generate_synthetic_shot(nt=500, nx=48, dt=0.002, dx=10):
#     """
#     生成合成炮集数据用于测试
#     """
#     t = np.arange(nt) * dt
#     x = np.arange(nx) * dx
    
#     data = np.zeros((nt, nx))
    
#     # 添加几个不同视速度的线性同相轴
#     velocities = [2000, 3000, -2500]  # m/s，负值表示反向传播
    
#     for v in velocities:
#         for i, offset in enumerate(x):
#             # 计算到达时间
#             t_arrival = abs(offset / v)
#             # 添加Ricker子波
#             freq = 25  # Hz
#             arrival_sample = int(t_arrival / dt)
#             if arrival_sample < nt:
#                 ricker = signal.ricker(100, 4)
#                 start = max(0, arrival_sample - 50)
#                 end = min(nt, arrival_sample + 50)
#                 wavelet_len = end - start
#                 data[start:end, i] += ricker[:wavelet_len] * 0.5
    
#     # 添加噪声
#     data += np.random.randn(nt, nx) * 0.05
    
#     return data, dt, dx


# # 加载原始数据（不需要手动切片）
# your_data = np.load(r'D:\桌面\面波压制\sh\wildamp_cs20.npy')[0]
# your_dt = 0.00025
# your_dx = 2
# save_path = r'D:\桌面\面波压制\sh\wildamp_cs20_fk'

# # 计算F-K谱：自动进行逐道归一化和截取前401个采样点
# fk, freqs, kx = fk_spectrum(your_data, your_dt, your_dx, 
#                              window=True, 
#                              freq_range=[0, 500], 
#                              save_path=save_path,
#                              normalize=True,      # 逐道归一化
#                              n_samples=None)       # 截取前401个采样点




import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def fk_spectrum(data, dt, dx, window=True, freq_range=None, k_range=None, save_path=None, 
                normalize=True, n_samples=None):
    # 数据预处理：逐道归一化
    if normalize:
        print("正在进行逐道归一化...")
        data_processed = np.zeros_like(data)
        for i in range(data.shape[1]):
            trace = data[:, i]
            trace_max = np.max(np.abs(trace))
            data_processed[:, i] = trace / trace_max if trace_max > 0 else trace
        data = data_processed
        print("逐道归一化完成")
    
    # 数据预处理：截取采样点
    if n_samples is not None and n_samples < data.shape[0]:
        print(f"截取前 {n_samples} 个采样点...")
        data = data[:n_samples, :]

    nt, nx = data.shape

    # 窗函数
    if window:
        win_t = np.hanning(nt).reshape(-1, 1)
        win_x = np.hanning(nx).reshape(1, -1)
        data_windowed = data * win_t * win_x
    else:
        data_windowed = data
    
    # 2D FFT
    fk = np.fft.fft2(data_windowed)
    fk = np.fft.fftshift(fk)
    fk_amp = np.abs(fk)
    fk_amp_db = 20 * np.log10(fk_amp + 1e-10)

    # freq & k 轴
    freqs = np.fft.fftfreq(nt, dt)
    freqs = np.fft.fftshift(freqs)
    kx = np.fft.fftfreq(nx, dx)
    kx = np.fft.fftshift(kx)

    # 图像绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：原始炮集
    extent_data = [0, nx * dx, nt * dt, 0]
    im1 = ax1.imshow(
        data, aspect='auto', cmap='seismic',
        extent=extent_data,
        vmin=-np.percentile(np.abs(data), 99),
        vmax=np.percentile(np.abs(data), 99)
    )
    ax1.set_xlabel('偏移距 (m)')
    ax1.set_ylabel('时间 (s)')
    ax1.set_title('原始炮集记录')
    plt.colorbar(im1, ax=ax1)

    # 右图：F-K 谱
    extent_fk = [kx.min(), kx.max(), freqs.max(), freqs.min()]
    im2 = ax2.imshow(
        fk_amp_db, aspect='auto', cmap='jet',
        extent=extent_fk, origin='upper'
    )
    ax2.set_xlabel('波数 (1/m)')
    ax2.set_ylabel('频率 (Hz)')
    ax2.set_title('F-K谱')
    ax2.axhline(y=0, color='w', linestyle='--', linewidth=0.5)
    ax2.axvline(x=0, color='w', linestyle='--', linewidth=0.5)

    if freq_range is not None:
        ax2.set_ylim([freq_range[1], freq_range[0]])
    else:
        ax2.set_ylim([freqs.max(), 0])

    if k_range is not None:
        ax2.set_xlim(k_range)
    else:
        ax2.set_xlim([kx.min(), kx.max()])

    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + "_fk.png", dpi=300)
        fk_data = {
            'fk_amp_db': fk_amp_db,
            'freqs': freqs,
            'kx': kx,
            'dt': dt,
            'dx': dx,
            'data_shape': data.shape
        }
        np.save(save_path + "_fk.npy", fk_data)
        print(f"已保存：{save_path}_fk.png 和 {save_path}_fk.npy")

    plt.close()

    return fk, freqs, kx


def batch_fk(folder_path, dt, dx, normalize=False, window=True):
    # 获取所有 .npy 文件
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    
    print(f"发现 {len(files)} 个npy文件，开始批量处理...\n")

    for f in files:
        file_path = os.path.join(folder_path, f)
        print(f"处理文件：{f}")

        # 加载数据
        data = np.load(file_path)

        # 如果数据是 (shot, nt, nx)，取第 0 个 shot
        if data.ndim == 3:
            data = data[0]

        # 保存路径（不带后缀）
        save_path = os.path.join(folder_path, f.replace(".npy", ""))

        fk_spectrum(
            data,
            dt,
            dx,
            window=window,
            freq_range=[0, 500],
            save_path=save_path,
            normalize=normalize,
            n_samples=None
        )

    print("\n全部处理完成！")


# -------------------------------
#            使用示例
# -------------------------------
folder = r"D:\桌面\面波压制\Field data shot65\npy数据"   # 你的文件夹路径
dt = 0.00025
dx = 3

batch_fk(folder, dt, dx)
