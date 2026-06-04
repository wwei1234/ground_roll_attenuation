import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import os
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False

# ============================================================
#  ★ 配置（修改这里）
# ============================================================
CONFIG = {
    "input_dir"    : r"最终数据",
    "output_dir"   : r"最终图件(增益2)",
    "dt"           : 0.004,              # 采样间隔 (s)
    "figsize"      : (5, 4),
    "dpi"          : 600,
    "formats"      : ["png", "eps", "pdf"],
    "cmap"         : "gray",
    "interpolation": "bicubic",
    "fontsize"     : 24,
 
    # ── 裁剪范围（填 None 表示不限制）──────────────────────────
    "time_range"   : (0, 3),         # 单位 s；None = 全部
    "trace_range"  : None, #(40, 120),          # 道号索引；None = 全部
    # ──────────────────────────────────────────────────────────

    # ── 色标模式 ───────────────────────────────────────────────
    # "fixed"      → 所有图统一使用 clim 的值，便于横向对比
    # "auto"       → 每张图自适应自身最大绝对值
    # "percentile" → 每张图取 ±第 clim_pct 百分位，抑制异常值
    "clim_mode"    : "fixed",
    "clim"         : (-1, 1),            # 仅 fixed 模式生效
    "clim_pct"     : 95,                 # 仅 percentile 模式生效
    # ──────────────────────────────────────────────────────────

    # ── 独立 colorbar 外观 ─────────────────────────────────────
    "colorbar_label"      : "Amplitude",  # colorbar 标签，填 "" 则不显示
    "colorbar_figsize"    : (1.2, 4),     # 单独 colorbar 图尺寸 (宽, 高)
    "colorbar_orientation": "vertical",   # "vertical" 或 "horizontal"
    "colorbar_nticks"     : 5,            # 刻度数量（含两端）
    # ──────────────────────────────────────────────────────────
}
# ============================================================


def crop_data(data: np.ndarray, cfg: dict):
    """按 time_range 和 trace_range 裁剪数据，返回裁剪后的 data 及实际时间/道号起止。"""
    n_samples, n_traces = data.shape
    dt = cfg["dt"]

    if cfg["time_range"] is not None:
        t_start, t_end = cfg["time_range"]
        i_start = max(0, int(round(t_start / dt)))
        i_end   = min(n_samples, int(round(t_end / dt)))
    else:
        i_start, i_end = 0, n_samples

    if cfg["trace_range"] is not None:
        tr_start, tr_end = cfg["trace_range"]
        tr_start = max(0, tr_start)
        tr_end   = min(n_traces, tr_end)
    else:
        tr_start, tr_end = 0, n_traces

    data_cropped = data[i_start:i_end, tr_start:tr_end]
    return data_cropped, i_start * dt, i_end * dt, tr_start, tr_end


def get_clim(data: np.ndarray, cfg: dict):
    """根据 clim_mode 计算色标范围。"""
    mode = cfg.get("clim_mode", "fixed")
    if mode == "fixed":
        return cfg["clim"]
    elif mode == "auto":
        vmax = np.max(np.abs(data))
        return (-vmax, vmax)
    elif mode == "percentile":
        pct  = cfg.get("clim_pct", 95)
        vmax = np.percentile(np.abs(data), pct)
        return (-vmax, vmax)
    else:
        raise ValueError(f"未知的 clim_mode: {mode}，请选择 fixed / auto / percentile")


def save_colorbar(clim: tuple, cfg: dict, fontsize: int, name: str = "colorbar"):
    """
    将独立的 colorbar 保存为与主图相同的格式文件。

    Parameters
    ----------
    clim     : (vmin, vmax) 色标范围
    cfg      : 全局配置字典
    fontsize : 字体大小
    name     : 输出文件名（不含扩展名），默认 "colorbar"
    """
    vmin, vmax = clim
    orientation = cfg.get("colorbar_orientation", "vertical")
    figsize     = cfg.get("colorbar_figsize", (1.2, 4))
    nticks      = cfg.get("colorbar_nticks", 5)
    label       = cfg.get("colorbar_label", "Amplitude")

    # 用 ScalarMappable 驱动 colorbar，无需绘制主图
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cfg["cmap"], norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_visible(False)  # 隐藏坐标轴，只保留 colorbar

    cb = fig.colorbar(
        sm,
        ax          = ax,
        orientation = orientation,
        fraction    = 0.9,   # colorbar 占 axes 的比例（这里 ax 已隐藏，fraction 控制宽度）
        pad         = 0.05,
    )

    # 均匀设置刻度
    ticks = np.linspace(vmin, vmax, nticks)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=fontsize)

    if label:
        cb.set_label(label, fontsize=fontsize)

    plt.tight_layout()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    for fmt in cfg["formats"]:
        save_path = os.path.join(cfg["output_dir"], f"{name}.{fmt}")
        fig.savefig(save_path, format=fmt, dpi=cfg["dpi"])
        print(f"  已保存 colorbar: {save_path}")

    plt.close(fig)


def plot_and_save(data: np.ndarray, name: str, cfg: dict, fontsize: int):
    data_plot, t0, t1, tr0, tr1 = crop_data(data, cfg)

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    im = ax.imshow(
        data_plot,
        interpolation = cfg["interpolation"],
        cmap          = cfg["cmap"],
        aspect        = "auto",
        extent        = [tr0, tr1, t1, t0],
    )
    im.set_clim(*get_clim(data_plot, cfg))

    ax.set_xlabel("Trace", fontsize=fontsize)
    ax.set_ylabel("Time / s", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
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
    fontsize  = cfg["fontsize"]

    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    if not npy_files:
        print(f"[!] 未找到任何 .npy 文件: {input_dir}")
        return

    print(f"共找到 {len(npy_files)} 个文件，开始批量绘图...\n")

    # 用于收集所有图的 clim（供 auto/percentile 模式在全部处理后统一保存 colorbar）
    collected_clim = None

    for fname in npy_files:
        fpath = os.path.join(input_dir, fname)
        name  = os.path.splitext(fname)[0]
        print(f"处理: {fname}")

        raw = np.load(fpath, allow_pickle=True)
        if raw.ndim == 0:
            obj = raw.item()
            if isinstance(obj, dict):
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

        n_samples, n_traces = data.shape
        print(f"  原始形状: {n_samples} 采样点 × {n_traces} 道")

        plot_and_save(data, name, cfg, fontsize)

        # 记录第一张图的 clim（auto/percentile 模式下各图 clim 可能不同，
        # 此处保存最后一张图的 clim；如需固定可改为只取第一张）
        data_plot, *_ = crop_data(data, cfg)
        collected_clim = get_clim(data_plot, cfg)

    # ── 保存独立 colorbar ──────────────────────────────────────
    # fixed 模式直接用配置值；其他模式用最后一张图的 clim
    if cfg.get("clim_mode", "fixed") == "fixed":
        cb_clim = cfg["clim"]
    else:
        cb_clim = collected_clim

    if cb_clim is not None:
        print("\n正在保存独立 colorbar...")
        save_colorbar(cb_clim, cfg, fontsize, name="colorbar")
    # ──────────────────────────────────────────────────────────

    print("\n全部完成！")


if __name__ == "__main__":
    main()