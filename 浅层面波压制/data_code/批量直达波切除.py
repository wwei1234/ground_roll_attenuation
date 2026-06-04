import numpy as np
import os
from pathlib import Path
from scipy import signal

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


# ========== 配置参数 ==========
input_folder  = r'npy文件汇总_filtered(10-150)'       # 输入文件夹
output_folder = r'npy文件汇总_filtered(10-150)_muted'  # 输出文件夹

# 直达波切除参数
center_trace = 19
velocity     = 1200
dt           = 0.00025
mute_time0   = 0
dx           = 2
taper        = 0
# ==============================


def process_folder(input_folder, output_folder,
                   center_trace, velocity, dt, mute_time0, dx, taper):
    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_path.glob('*.npy'))
    if not npy_files:
        print(f"在 {input_folder} 中未找到 .npy 文件")
        return

    print(f"共找到 {len(npy_files)} 个 .npy 文件")
    print(f"直达波切除参数: center_trace={center_trace}, velocity={velocity}, "
          f"dt={dt}, mute_time0={mute_time0}, dx={dx}, taper={taper}")
    print(f"输出文件夹: {output_path}\n")

    success, failed = 0, []

    for idx, file_path in enumerate(npy_files, 1):
        print(f"[{idx}/{len(npy_files)}] 处理: {file_path.name}", end=" ... ")
        try:
            data = np.load(file_path)

            result = remove_direct_wave(
                data,
                center_trace=center_trace,
                velocity=velocity,
                dt=dt,
                mute_time0=mute_time0,
                dx=dx,
                taper=taper,
            )

            np.save(output_path / file_path.name, result)
            print(f"完成  shape={data.shape}")
            success += 1

        except Exception as e:
            print(f"失败！错误: {e}")
            failed.append((file_path.name, str(e)))

    print(f"\n{'='*50}")
    print(f"处理完成: 成功 {success} 个，失败 {len(failed)} 个")
    if failed:
        print("失败文件:")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == '__main__':
    process_folder(
        input_folder  = input_folder,
        output_folder = output_folder,
        center_trace  = center_trace,
        velocity      = velocity,
        dt            = dt,
        mute_time0    = mute_time0,
        dx            = dx,
        taper         = taper,
    )