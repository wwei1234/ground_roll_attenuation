import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import segyio
import os
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False

# ============================================================
#  ★ 配置
# ============================================================
CONFIG = {
    "output_dir"   : r"图片汇总\合成记录最终使用图",
    "dx"           : 1,
    "dz"           : 1,
    "max_samples"  : None,
    "figsize_w"    : 6,       # 只控制宽度，高度自动计算
    "dpi"          : 600,
    "formats"      : ["png", "eps", "pdf"],
    "aspect"       : 4,

    # ★ 离散值 → 颜色映射（按实际数值修改）
    "value_colors" : {
        800: "#D0941B",
        1500: "#1952C4",
        2000: "#55248D",
    },
}
# ============================================================

def read_segy(data_dir, shotnum=0):
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)
        if shotnum:
            shot_num = shotnum
        else:
            shot_num = len(set(sourceX))
        len_shot = trace_num // shot_num
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num, time, len_shot))
        for j in range(0, shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
        return data


def make_discrete_cmap(value_colors: dict):
    """根据 {值: 颜色} 字典构建离散 colormap 和 norm"""
    values = sorted(value_colors.keys())
    colors = [value_colors[v] for v in values]

    cmap = mcolors.ListedColormap(colors)

    bounds = (
        [values[0] - (values[1] - values[0]) / 2]
        + [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]
        + [values[-1] + (values[-1] - values[-2]) / 2]
    )
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, values, colors


def plot_and_save(data: np.ndarray, name: str, cfg: dict):
    data = data[:, 0:400]
    nz, nx = data.shape
    dx, dz = cfg["dx"], cfg["dz"]
    x_start, x_end = 0, (nx - 1) * dx
    z_start, z_end = 0, (nz - 1) * dz

    cmap, norm, values, colors = make_discrete_cmap(cfg["value_colors"])

    # 根据实际数据比例自动推算 figsize，避免大量空白
    fig_w = cfg["figsize_w"]
    data_ratio = (nz * dz) / (nx * dx)
    fig_h = fig_w * data_ratio * cfg["aspect"]
    fig_h = max(3.0, min(fig_h, 14.0))  # 限制在合理范围内

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        aspect=cfg["aspect"],
        extent=[x_start, x_end, z_end, z_start],
        interpolation="nearest",
    )

    cbar = fig.colorbar(
        im, ax=ax,
        ticks=values,
        fraction=0.02,
        pad=0.03,
        shrink=0.5,
    )
    cbar.ax.set_yticklabels([str(v) for v in values], fontsize=14)
    cbar.set_label("Velocity / (m/s)", fontsize=14)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel("Distance / m", fontsize=16)
    ax.set_ylabel("Depth / m",    fontsize=16)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    # 如果还有残余空白，取消注释下一行并微调数值
    # fig.subplots_adjust(top=0.93, bottom=0.04)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    for fmt in cfg["formats"]:
        save_path = os.path.join(cfg["output_dir"], f"{name}.{fmt}")
        fig.savefig(save_path, format=fmt, dpi=cfg["dpi"])
        print(f"  已保存: {save_path}")

    plt.close(fig)

# ============================================================
#  主流程
# ============================================================
data = read_segy(r"训练数据\cs_free+Model-Complex.sgy", shotnum=1)
print(data.shape)

unique_vals = np.unique(data[0])
print("模型中的唯一值：", unique_vals)

cfg = CONFIG
plot_and_save(data[0], "shot_0", cfg)