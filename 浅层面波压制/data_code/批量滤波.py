import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from pathlib import Path

# ========== 配置参数 ==========
input_folder = r'Synthetic data test 2\npy文件汇总'   # 输入文件夹
output_folder = r'Synthetic data test 2\npy文件汇总_filtered(10-150)'  # 输出文件夹

# 滤波参数
sample_rate = 4000  # 采样率 (Hz)
low_freq = 10       # 低频截止 (Hz)
high_freq = 150     # 高频截止 (Hz)
filter_order = 5    # 滤波器阶数
# ==============================


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    带通滤波器
    参数:
        data: 输入数据，shape为 (时间采样点, 道数)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        fs: 采样率 (Hz)
        order: 滤波器阶数
    返回:
        filtered_data: 滤波后的数据
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = signal.sosfiltfilt(sos, data[:, i])

    return filtered_data


def process_folder(input_folder, output_folder, low_freq, high_freq, sample_rate, filter_order):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)

    # 找到所有 .npy 文件
    npy_files = sorted(input_path.glob('*.npy'))

    if not npy_files:
        print(f"在 {input_folder} 中未找到 .npy 文件")
        return

    print(f"共找到 {len(npy_files)} 个 .npy 文件")
    print(f"滤波参数: {low_freq}-{high_freq} Hz，采样率: {sample_rate} Hz，阶数: {filter_order}")
    print(f"输出文件夹: {output_path}\n")

    success, failed = 0, []

    for idx, file_path in enumerate(npy_files, 1):
        print(f"[{idx}/{len(npy_files)}] 处理: {file_path.name}", end=" ... ")

        try:
            data = np.load(file_path)

            # 兼容 1D 数据（单道）
            if data.ndim == 1:
                data = data[:, np.newaxis]
                squeezed = True
            else:
                squeezed = False

            filtered_data = bandpass_filter(data, low_freq, high_freq, sample_rate, filter_order)

            if squeezed:
                filtered_data = filtered_data.squeeze()

            # 输出文件名保持不变
            out_file = output_path / file_path.name
            np.save(out_file, filtered_data)

            print(f"完成  shape={data.shape}")
            success += 1

        except Exception as e:
            print(f"失败！错误: {e}")
            failed.append((file_path.name, str(e)))

    # 汇总
    print(f"\n{'='*50}")
    print(f"处理完成: 成功 {success} 个，失败 {len(failed)} 个")
    if failed:
        print("失败文件:")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == '__main__':
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        low_freq=low_freq,
        high_freq=high_freq,
        sample_rate=sample_rate,
        filter_order=filter_order,
    )