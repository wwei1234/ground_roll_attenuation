import os
import numpy as np
from scipy import signal

def fk_spectrum(data, dt, dx, window=True, freq_range=None, k_range=None, save_path=None, 
                normalize=True, n_samples=None):
    data = data[:, 50:120]
    if normalize:
        data_processed = np.zeros_like(data)
        for i in range(data.shape[1]):
            trace = data[:, i]
            trace_max = np.max(np.abs(trace))
            data_processed[:, i] = trace / trace_max if trace_max > 0 else trace
        data = data_processed
    
    if n_samples is not None and n_samples < data.shape[0]:
        data = data[:n_samples, :]

    nt, nx = data.shape

    if window:
        win_t = np.hanning(nt).reshape(-1, 1)
        win_x = np.hanning(nx).reshape(1, -1)
        data_windowed = data * win_t * win_x
    else:
        data_windowed = data
    
    fk = np.fft.fft2(data_windowed)
    fk = np.fft.fftshift(fk)
    fk_amp = np.abs(fk)
    fk_amp_db = 20 * np.log10(fk_amp + 1e-10)

    freqs = np.fft.fftfreq(nt, dt)
    freqs = np.fft.fftshift(freqs)
    kx = np.fft.fftfreq(nx, dx)
    kx = np.fft.fftshift(kx)

    if save_path is not None:
        np.save(save_path + "_fk.npy", {
            'fk_amp_db': fk_amp_db,
            'freqs': freqs,
            'kx': kx,
            'dt': dt,
            'dx': dx,
            'data_shape': data.shape
        })
        print(f"已保存：{save_path}_fk.npy")

    return fk, freqs, kx


def batch_fk(input_folder, output_folder, dt, dx, normalize=False, window=True):
    os.makedirs(output_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
    print(f"发现 {len(files)} 个npy文件，开始批量处理...\n")

    for f in files:
        file_path = os.path.join(input_folder, f)
        print(f"处理文件：{f}")

        data = np.load(file_path)
        if data.ndim == 3:
            data = data[0]

        save_path = os.path.join(output_folder, f.replace(".npy", ""))

        fk_spectrum(
            data, dt, dx,
            window=window,
            save_path=save_path,
            normalize=normalize,
            n_samples=None
        )

    print("\n全部处理完成！")


# -------------------------------
#            使用示例
# -------------------------------
input_folder  = r"最终数据_直达波切除"
output_folder = r"F-K数据_直达波切除"
dt = 0.004
dx = 25

batch_fk(input_folder, output_folder, dt, dx)