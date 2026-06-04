# import numpy as np
# import segyio

# # ===================== #
# # 1. 读取 npy 数据
# # ===================== #
# npy_file = r"D:\桌面\面波压制\Synthetic data test2\npy文件汇总\noised2_1127.npy"
# data = np.load(npy_file)   # data.shape = (nsamples, ntraces)
# nsamples, ntraces = data.shape
# print(f"数据维度: {nsamples} 个采样点, {ntraces} 道")

# # ===================== #
# # 2. 设置基本参数
# # ===================== #
# sgy_file = r"D:\桌面\面波压制\Synthetic data test2\输入sgy文件\noised2_1127.sgy"

# dt = 250  # 采样间隔 (微秒)，需要根据实际情况修改
# iline = 0  # inline 起始编号，可自定义
# xline = 0  # crossline 起始编号，可自定义

# # ===================== #
# # 3. 创建 SGY 文件
# # ===================== #
# spec = segyio.spec()
# spec.format = 5                # IEEE 浮点格式 (常用)
# spec.samples = range(nsamples) # 每道采样点数
# spec.tracecount = ntraces      # 道数

# with segyio.create(sgy_file, spec) as f:
#     for i in range(ntraces):
#         # 写入道数据
#         f.trace[i] = data[:, i]

#         # ===================== #
#         # 设置道头信息
#         # ===================== #
#         f.header[i] = {
#             segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,   # 道号（从1开始）
#             segyio.TraceField.FieldRecord: 1,               # 炮号
#             segyio.TraceField.TraceNumber: i + 1,           # 道号
#             segyio.TraceField.CDP: i + 1,                   # CDP编号
#             segyio.TraceField.offset: i * 25,               # ✅ 偏移距（小写 offset）
#             segyio.TraceField.INLINE_3D: iline,             # inline编号
#             segyio.TraceField.CROSSLINE_3D: xline + i,      # crossline编号
#             segyio.TraceField.TRACE_SAMPLE_COUNT: nsamples, # 采样点数
#             segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt     # 采样间隔
#         }

#     # ===================== #
#     # 设置二进制头信息
#     # ===================== #
#     f.bin[segyio.BinField.Traces] = ntraces
#     f.bin[segyio.BinField.Samples] = nsamples
#     f.bin[segyio.BinField.Interval] = dt

# print(f"✅ SGY 文件已保存: {sgy_file}")

















import numpy as np
import segyio
import os

# ============================
# 1. 输入与输出路径（自行修改）
# ============================
input_folder = r"最终数据"
output_folder = r"最终数据_sgy"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================
# 2. 基本参数
# ============================
dt = 4000                     # 微秒
delay_ms = 0               # 起始时间 0.025 s = 25 ms
iline = 0
xline_start = 0
offset_step = 25


# ============================
# 3. 定义转换函数
# ============================
def convert_npy_to_sgy(npy_path, sgy_path):
    data = np.load(npy_path)
    nsamples, ntraces = data.shape

    print(f"处理文件：{os.path.basename(npy_path)}")
    print(f"  数据维度: {nsamples} x {ntraces}")

    spec = segyio.spec()
    spec.format = 5
    spec.samples = range(nsamples)
    spec.tracecount = ntraces

    with segyio.create(sgy_path, spec) as f:

        for i in range(ntraces):
            f.trace[i] = data[:, i]

            # 设置 trace header
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.FieldRecord: 1,
                segyio.TraceField.TraceNumber: i + 1,
                segyio.TraceField.CDP: i + 1,
                segyio.TraceField.offset: i * offset_step,
                segyio.TraceField.INLINE_3D: iline,
                segyio.TraceField.CROSSLINE_3D: xline_start + i,
                segyio.TraceField.TRACE_SAMPLE_COUNT: nsamples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt,

                # ⭐ 加入起始记录时间（单位必须为 ms）
                segyio.TraceField.DelayRecordingTime: delay_ms,
            }

        f.bin[segyio.BinField.Traces] = ntraces
        f.bin[segyio.BinField.Samples] = nsamples
        f.bin[segyio.BinField.Interval] = dt

    print(f"  ✅ 输出：{sgy_path}\n")


# ============================
# 4. 批处理所有 npy 文件
# ============================
for file in os.listdir(input_folder):
    if file.lower().endswith(".npy"):
        npy_file = os.path.join(input_folder, file)
        sgy_file = os.path.join(output_folder, file.replace(".npy", ".sgy"))
        convert_npy_to_sgy(npy_file, sgy_file)

print("🎉 所有 npy 文件已全部转换完成！")