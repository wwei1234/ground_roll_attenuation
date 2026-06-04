import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  ★ 配置（修改这里）
# ============================================================
CONFIG = {
    "input_folder"  : r"数据汇总\合成记录\源数据汇总_filtered(20-150)_muted",  # 输入 npy 文件夹
    "output_folder" : r"数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk",       # 输出文件夹（与输入分离）
    "dt"            : 0.00025,    # 时间采样间隔（秒）
    "dx"            : 2,          # 道间距（米）
    "normalize"     : False,      # 是否逐道归一化
    "window"        : True,       # 是否加窗
    "freq_range"    : [0, 500],   # 频率显示范围（Hz），None 表示不限制
    "n_samples"     : None,       # 截取采样点数，None 表示不截取
    "save_png"      : False,       # 保存 F-K 图
    "save_npy"      : True,       # 保存 F-K 数据
}
# ============================================================


def fk_spectrum(data, dt, dx, window=True, freq_range=None,
                normalize=True, n_samples=None,
                output_folder=None, file_stem=None,
                save_png=True, save_npy=True):
    """
    计算单炮 F-K 谱并保存。

    Parameters
    ----------
    data          : (nt, nx) 2D array
    dt            : 时间采样间隔（秒）
    dx            : 道间距（米）
    output_folder : 输出文件夹路径
    file_stem     : 输出文件名（不含扩展名）
    """
    # 逐道归一化
    if normalize:
        data_processed = np.zeros_like(data)
        for i in range(data.shape[1]):
            trace = data[:, i]
            trace_max = np.max(np.abs(trace))
            data_processed[:, i] = trace / trace_max if trace_max > 0 else trace
        data = data_processed

    # 截取采样点
    if n_samples is not None and n_samples < data.shape[0]:
        data = data[:n_samples, :]

    nt, nx = data.shape

    # 窗函数
    if window:
        data_windowed = data * np.hanning(nt).reshape(-1, 1) \
                              * np.hanning(nx).reshape(1, -1)
    else:
        data_windowed = data

    # 2D FFT
    fk        = np.fft.fftshift(np.fft.fft2(data_windowed))
    fk_amp_db = 20 * np.log10(np.abs(fk) + 1e-10)

    freqs = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    kx    = np.fft.fftshift(np.fft.fftfreq(nx, dx))

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    vmax = np.percentile(np.abs(data), 99)
    im1 = ax1.imshow(data, aspect='auto', cmap='seismic',
                     extent=[0, nx * dx, nt * dt, 0],
                     vmin=-vmax, vmax=vmax)
    ax1.set_xlabel('偏移距 (m)')
    ax1.set_ylabel('时间 (s)')
    ax1.set_title('原始炮集记录')
    plt.colorbar(im1, ax=ax1)

    extent_fk = [kx.min(), kx.max(), freqs.max(), freqs.min()]
    im2 = ax2.imshow(fk_amp_db, aspect='auto', cmap='jet',
                     extent=extent_fk, origin='upper')
    ax2.set_xlabel('波数 (1/m)')
    ax2.set_ylabel('频率 (Hz)')
    ax2.set_title('F-K 谱')
    ax2.axhline(y=0, color='w', linestyle='--', linewidth=0.5)
    ax2.axvline(x=0, color='w', linestyle='--', linewidth=0.5)

    if freq_range is not None:
        ax2.set_ylim([freq_range[1], freq_range[0]])
    else:
        ax2.set_ylim([freqs.max(), 0])
    ax2.set_xlim([kx.min(), kx.max()])

    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()

    # 保存
    if output_folder and file_stem:
        if save_png:
            png_path = os.path.join(output_folder, f"{file_stem}_fk.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"  PNG: {png_path}")
        if save_npy:
            npy_path = os.path.join(output_folder, f"{file_stem}_fk.npy")
            np.save(npy_path, {
                'fk_amp_db'  : fk_amp_db,
                'freqs'      : freqs,
                'kx'         : kx,
                'dt'         : dt,
                'dx'         : dx,
                'data_shape' : data.shape,
            })
            print(f"  NPY: {npy_path}")

    plt.close(fig)
    return fk, freqs, kx


def batch_fk(input_folder, output_folder, dt, dx,
             normalize=False, window=True, freq_range=None,
             n_samples=None, save_png=True, save_npy=True):

    os.makedirs(output_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
    if not files:
        print(f"[!] 未找到任何 .npy 文件: {input_folder}")
        return

    print(f"输入目录 : {input_folder}")
    print(f"输出目录 : {output_folder}")
    print(f"共找到 {len(files)} 个文件，开始处理...\n")

    for i, fname in enumerate(files, 1):
        file_path = os.path.join(input_folder, fname)
        file_stem = os.path.splitext(fname)[0]
        print(f"[{i}/{len(files)}] {fname}")

        data = np.load(file_path)
        if data.ndim == 3:
            data = data[0]   # (shot, nt, nx) → 取第 0 炮
        if data.ndim != 2:
            print(f"  [跳过] 数据维度异常: {data.shape}")
            continue

        print(f"  形状: {data.shape[0]} 采样点 × {data.shape[1]} 道")

        fk_spectrum(
            data          = data,
            dt            = dt,
            dx            = dx,
            window        = window,
            freq_range    = freq_range,
            normalize     = normalize,
            n_samples     = n_samples,
            output_folder = output_folder,
            file_stem     = file_stem,
            save_png      = save_png,
            save_npy      = save_npy,
        )

    print("\n全部处理完成！")


# ============================================================
#  主程序
# ============================================================
if __name__ == "__main__":
    cfg = CONFIG
    batch_fk(
        input_folder  = cfg["input_folder"],
        output_folder = cfg["output_folder"],
        dt            = cfg["dt"],
        dx            = cfg["dx"],
        normalize     = cfg["normalize"],
        window        = cfg["window"],
        freq_range    = cfg["freq_range"],
        n_samples     = cfg["n_samples"],
        save_png      = cfg["save_png"],
        save_npy      = cfg["save_npy"],
    )