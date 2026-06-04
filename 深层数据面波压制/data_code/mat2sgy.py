import h5py
import segyio
import numpy as np

# === 读取 v7.3 版本的 .mat 文件 ===
mat_path = r"D:\桌面\面波压制\Raw_data.mat"  # 你的 .mat 文件路径
with h5py.File(mat_path, 'r') as f:
    data = np.array(f['data']).T       # 注意：h5py读出来是转置的 (道在行)，需转置回来
    dt = float(np.array(f['dt']))      # 采样间隔（秒）
    offset = np.array(f['offset']).squeeze()

# === 构造 SEGY 文件参数 ===
n_samples, n_traces = data.shape
spec = segyio.spec()
spec.sorting = 1
spec.format = 1  # 1=IBM浮点；5=IEEE浮点（部分软件更兼容）
spec.samples = np.arange(n_samples) * dt
spec.ilines = np.arange(1, n_traces + 1)
spec.xlines = np.arange(1, 2)

# === 写出 SEGY 文件 ===
segy_path = r"D:\桌面\面波压制\Raw_data.sgy"
with segyio.create(segy_path, spec) as f:
    for i in range(n_traces):
        f.trace[i] = data[:, i]
        f.header[i] = {
            segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
            segyio.TraceField.offset: int(offset[i])
        }

    # 二进制头部分
    f.bin[segyio.BinField.Interval] = int(dt * 1e6)  # 单位: 微秒
    f.flush()

print(f"✅ 已生成 SEGY 文件: {segy_path}")
