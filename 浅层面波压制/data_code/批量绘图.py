import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False

# ============================================================
#  ★ 配置（修改这里）
# ============================================================
CONFIG = {
    "input_dir"    : r"Synthetic data test 2\mark",        # 输入 npy 文件夹
    "output_dir"   : r"Synthetic data test 2\图片汇总",        # 输出图片文件夹
    "dt"           : 0.25,                  # 采样间隔 (ms)
    "max_samples"  : None,                # 截取时间采样点数，None 表示不截取
    "clim"         : (-1, 1),            # 色标范围
    "figsize"      : (3, 4),             # 图像尺寸（英寸）
    "dpi"          : 600,                # 保存分辨率
    "formats"      : ["png", "eps", "pdf"],            # 保存格式，可选 "png" "eps" "pdf"
    "cmap"         : "seismic",          # 色图
    "interpolation": "bicubic",          # 插值方式
}
# ============================================================

def plot_and_save(data: np.ndarray, name: str, cfg: dict):
    # data = data[100:500, 5:43]  # 截取部分数据进行绘图
    n_samples, n_traces = data.shape
    dt = cfg["dt"]
    time_axis = np.arange(n_samples) * dt + 25

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    im = ax.imshow(
        data,
        interpolation = cfg["interpolation"],
        cmap          = cfg["cmap"],
        aspect        = "auto",
        extent        = [0, n_traces, time_axis[-1], time_axis[0]],
    )
    im.set_clim(*cfg["clim"])
    ax.set_xlabel("Trace", fontsize=16)
    ax.set_ylabel("Time / ms", fontsize=16)
    ax.tick_params(labelsize=16)
    plt.tight_layout()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    for fmt in cfg["formats"]:
        save_path = os.path.join(cfg["output_dir"], f"{name}.{fmt}")
        fig.savefig(save_path, format=fmt, dpi=cfg["dpi"])
        print(f"  已保存: {save_path}")

    plt.close(fig)


def main():
    cfg       = CONFIG
    input_dir = cfg["input_dir"]

    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    if not npy_files:
        print(f"[!] 未找到任何 .npy 文件: {input_dir}")
        return

    print(f"共找到 {len(npy_files)} 个文件，开始批量绘图...\n")

    for fname in npy_files:
        fpath = os.path.join(input_dir, fname)
        name  = os.path.splitext(fname)[0]
        print(f"处理: {fname}")

        raw = np.load(fpath, allow_pickle=True)
        # 兼容两种存储格式：普通 ndarray 或 pickle 字典（如 F-K 谱数据）
        if raw.ndim == 0:
            obj = raw.item()
            if isinstance(obj, dict):
                # 字典格式：取第一个二维数组字段
                data = None
                for v in obj.values():
                    if isinstance(v, np.ndarray) and v.ndim == 2:
                        data = v.astype(np.float32)
                        break
                if data is None:
                    print(f"  [跳过] 字典中未找到 2D 数组，keys={list(obj.keys())}")
                    continue
            else:
                print(f"  [跳过] 不支持的数据类型: {type(obj)}")
                continue
        else:
            data = raw.astype(np.float32)

        if data.ndim != 2:
            print(f"  [跳过] 数据维度不是 2D，shape={data.shape}")
            continue

        if cfg["max_samples"] is not None:
            data = data[:cfg["max_samples"], :]

        n_samples, n_traces = data.shape
        print(f"  形状: {n_samples} 采样点 × {n_traces} 道")

        plot_and_save(data, name, cfg)

    print("\n全部完成！")


if __name__ == "__main__":
    main()