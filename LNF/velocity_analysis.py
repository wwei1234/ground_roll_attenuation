"""
交互式面波速度拾取工具
=======================
使用方法：
    python velocity_picker.py                        # 自动读取 original.npy
    python velocity_picker.py path/to/data.npy       # 指定数据文件
    python velocity_picker.py path/to/data.npy --dt 0.004 --dx 25

操作说明：
    - 在炮集图上【按住左键拖动】画一条线 → 自动计算并显示视速度
    - 右键点击已有线条 → 删除该线条
    - 按 C 键 → 清除所有线条
    - 按 S 键 → 保存当前拾取结果到 velocity_picks.txt
    - 按 F 键 → 显示/隐藏 F-K 频谱（帮助判断面波频率范围）
    - 按 R 键 → 重置视图缩放
    - 关闭窗口后自动保存

依赖：pip install numpy matplotlib scipy
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # Windows 首选；若报错改为 "Qt5Agg" 或 "WXAgg"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.ndimage import convolve1d


# ============================================================
# 颜色方案
# ============================================================
COLORS = [
    "#FF4444", "#44AAFF", "#44FF88", "#FFB344", "#CC44FF",
    "#FF44CC", "#44FFEE", "#FFFF44", "#FF8844", "#88FF44",
]

PANEL_BG    = "#0D1117"
AXES_BG     = "#161B22"
TEXT_COLOR  = "#E6EDF3"
GRID_COLOR  = "#30363D"
ACCENT      = "#58A6FF"


# ============================================================
# 数据预处理
# ============================================================

def load_data(path):
    data = np.load(str(path)).astype(np.float64).squeeze()
    if data.ndim != 2:
        raise ValueError(f"期望二维数组，实际形状：{data.shape}")
    return data


def compute_fk(data, dt, dx):
    """计算 F-K 频谱（振幅谱取 dB）。"""
    nt, nx = data.shape
    nfft_t = max(512, int(2 ** np.ceil(np.log2(nt * 2))))
    nfft_x = max(64,  int(2 ** np.ceil(np.log2(nx * 2))))

    FK = np.fft.fftshift(
        np.abs(np.fft.fft2(data, s=(nfft_t, nfft_x)))
    )
    FK_db = 20 * np.log10(FK / (FK.max() + 1e-10) + 1e-6)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft_t, d=dt))
    kwave = np.fft.fftshift(np.fft.fftfreq(nfft_x, d=dx))
    return FK_db, freqs, kwave


def estimate_amplitude_scale(data):
    """用第 95 百分位振幅做显示裁切，避免异常道影响色标。"""
    return np.percentile(np.abs(data), 95)


# ============================================================
# 核心：速度拾取器
# ============================================================

class VelocityPicker:

    def __init__(self, data, dt=0.004, dx=25, title="炮集速度拾取"):
        self.data  = data
        self.dt    = dt
        self.dx    = dx
        self.title = title
        self.nt, self.nx = data.shape

        # 时间轴和偏移距轴
        self.t_axis = np.arange(self.nt) * dt          # 单位 s
        self.x_axis = np.arange(self.nx) * dx          # 单位 m

        # 拾取记录
        self.picks      = []   # list of dict: {x0,t0,x1,t1,v,color,artists}
        self.color_idx  = 0

        # 拖线状态
        self._drag_start  = None   # (x_data, t_data)
        self._temp_line   = None
        self._dragging    = False

        # F-K 面板是否显示
        self._fk_visible  = False
        self._fk_computed = False

        self._build_layout()
        self._connect_events()
        self._draw_seismic()
        self._update_info_panel()

    # ----------------------------------------------------------
    # 布局
    # ----------------------------------------------------------

    def _build_layout(self):
        plt.rcParams.update({
            "figure.facecolor":  PANEL_BG,
            "axes.facecolor":    AXES_BG,
            "axes.edgecolor":    GRID_COLOR,
            "axes.labelcolor":   TEXT_COLOR,
            "xtick.color":       TEXT_COLOR,
            "ytick.color":       TEXT_COLOR,
            "text.color":        TEXT_COLOR,
            "grid.color":        GRID_COLOR,
            "grid.linestyle":    "--",
            "grid.alpha":        0.4,
            "font.family":       "monospace",
        })

        self.fig = plt.figure(figsize=(16, 9), facecolor=PANEL_BG)
        self.fig.canvas.manager.set_window_title("LNF 面波速度拾取工具")

        # 主布局：左侧炮集 | 右侧面板
        gs_main = gridspec.GridSpec(
            1, 2, figure=self.fig,
            width_ratios=[3, 1],
            left=0.05, right=0.98,
            top=0.95, bottom=0.07,
            wspace=0.04,
        )

        # 左侧：炮集 + 可选 F-K（竖向堆叠）
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_main[0],
            height_ratios=[3, 1], hspace=0.06,
        )
        self.ax_seis = self.fig.add_subplot(gs_left[0])
        self.ax_fk   = self.fig.add_subplot(gs_left[1])
        self.ax_fk.set_visible(False)

        # 右侧面板：速度列表 + 操作说明
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_main[1],
            height_ratios=[3, 1], hspace=0.1,
        )
        self.ax_info  = self.fig.add_subplot(gs_right[0])
        self.ax_help  = self.fig.add_subplot(gs_right[1])

        for ax in [self.ax_info, self.ax_help]:
            ax.set_xticks([]); ax.set_yticks([])

        self._draw_help_panel()

    # ----------------------------------------------------------
    # 炮集绘制
    # ----------------------------------------------------------

    def _draw_seismic(self):
        ax = self.ax_seis
        ax.cla()
        vmax = estimate_amplitude_scale(self.data)

        ax.imshow(
            self.data,
            aspect="auto",
            cmap="seismic",
            vmin=-vmax, vmax=vmax,
            extent=[
                self.x_axis[0]  - self.dx / 2,
                self.x_axis[-1] + self.dx / 2,
                self.t_axis[-1] + self.dt / 2,
                self.t_axis[0]  - self.dt / 2,
            ],
            interpolation="bilinear",
        )

        ax.set_xlabel("偏移距 (m)",  fontsize=11, labelpad=6)
        ax.set_ylabel("时间 (s)",    fontsize=11, labelpad=6)
        ax.set_title(self.title,     fontsize=13, color=ACCENT, pad=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(self.x_axis[0]  - self.dx / 2,
                    self.x_axis[-1] + self.dx / 2)
        ax.set_ylim(self.t_axis[-1] + self.dt / 2,
                    self.t_axis[0]  - self.dt / 2)

        # 重绘已有拾取线
        for pick in self.picks:
            self._draw_pick_artists(pick)

    # ----------------------------------------------------------
    # F-K 面板
    # ----------------------------------------------------------

    def _draw_fk(self):
        if not self._fk_computed:
            self._fk_db, self._fk_freqs, self._fk_k = compute_fk(
                self.data, self.dt, self.dx
            )
            self._fk_computed = True

        ax = self.ax_fk
        ax.cla()
        f_max = 1 / (2 * self.dt)
        k_max = 1 / (2 * self.dx)

        mask_f = (self._fk_freqs >= 0) & (self._fk_freqs <= f_max)
        fk_show = self._fk_db[mask_f, :]

        ax.imshow(
            fk_show,
            aspect="auto",
            cmap="inferno",
            vmin=-60, vmax=0,
            extent=[-k_max, k_max, f_max, 0],
            interpolation="bilinear",
        )
        ax.set_xlabel("波数 (1/m)", fontsize=9)
        ax.set_ylabel("频率 (Hz)", fontsize=9)
        ax.set_title("F-K 频谱（按 F 关闭）", fontsize=9, color=ACCENT)
        ax.set_ylim(0, min(f_max, 80))
        ax.grid(True, alpha=0.2)

        # 画几条等速度线供参考
        for v_ref, lc in [(200,"#88ff88"), (500,"#ffff44"), (1000,"#ff8844")]:
            k_line = np.linspace(-k_max, k_max, 200)
            f_line = np.abs(k_line) * v_ref
            ax.plot(k_line, f_line, color=lc, lw=0.8, alpha=0.6,
                    label=f"{v_ref} m/s")
        ax.legend(fontsize=7, loc="upper right",
                  facecolor=AXES_BG, edgecolor=GRID_COLOR)

    # ----------------------------------------------------------
    # 信息面板
    # ----------------------------------------------------------

    def _update_info_panel(self):
        ax = self.ax_info
        ax.cla()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(AXES_BG)
        ax.set_title("拾取结果", fontsize=11, color=ACCENT, pad=8)

        if not self.picks:
            ax.text(0.5, 0.55,
                    "尚无拾取\n\n在炮集上\n按住左键拖动\n画线测速",
                    ha="center", va="center",
                    fontsize=11, color=GRID_COLOR,
                    transform=ax.transAxes,
                    linespacing=2.0)
        else:
            lines = ["  #    速度 (m/s)   ΔX (m)  ΔT (ms)"]
            lines.append("─" * 38)
            for i, p in enumerate(self.picks):
                dx_m  = abs(p["x1"] - p["x0"])
                dt_ms = abs(p["t1"] - p["t0"]) * 1000
                v     = p["v"]
                marker = "→" if v > 0 else "←"
                lines.append(
                    f"  {i+1:2d}  {marker} {abs(v):8.1f}    "
                    f"{dx_m:5.0f}   {dt_ms:6.1f}"
                )

            # 统计
            vs = [abs(p["v"]) for p in self.picks]
            lines.append("─" * 38)
            lines.append(f"  最小: {min(vs):.1f} m/s")
            lines.append(f"  最大: {max(vs):.1f} m/s")
            lines.append(f"  均值: {np.mean(vs):.1f} m/s")
            lines.append("")
            lines.append(f"  共 {len(self.picks)} 条")

            ax.text(0.05, 0.97, "\n".join(lines),
                    ha="left", va="top",
                    fontsize=9, color=TEXT_COLOR,
                    transform=ax.transAxes,
                    fontfamily="monospace",
                    linespacing=1.6)

    def _draw_help_panel(self):
        ax = self.ax_help
        ax.cla()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(AXES_BG)
        ax.set_title("快捷键", fontsize=10, color=ACCENT, pad=6)

        helps = [
            ("左键拖动", "画线测速"),
            ("右键点线", "删除线条"),
            ("C",        "清除全部"),
            ("S",        "保存结果"),
            ("F",        "F-K 频谱"),
            ("R",        "重置视图"),
        ]
        txt = "\n".join(f"  {k:<10s} {v}" for k, v in helps)
        ax.text(0.05, 0.92, txt,
                ha="left", va="top",
                fontsize=9, color=TEXT_COLOR,
                transform=ax.transAxes,
                fontfamily="monospace",
                linespacing=1.8)

    # ----------------------------------------------------------
    # 绘制单条拾取线
    # ----------------------------------------------------------

    def _draw_pick_artists(self, pick):
        """在炮集轴上画拾取线 + 标注速度，并记录 artist 列表。"""
        ax   = self.ax_seis
        col  = pick["color"]
        x0, t0, x1, t1 = pick["x0"], pick["t0"], pick["x1"], pick["t1"]
        v    = pick["v"]

        # 主线
        line, = ax.plot([x0, x1], [t0, t1],
                        color=col, lw=2.0, alpha=0.9,
                        solid_capstyle="round", zorder=5)

        # 端点圆点
        dot0, = ax.plot(x0, t0, "o", color=col, ms=6, zorder=6)
        dot1, = ax.plot(x1, t1, "o", color=col, ms=6, zorder=6)

        # 速度标注
        xm = (x0 + x1) / 2
        tm = (t0 + t1) / 2
        sign = "+" if v >= 0 else ""
        label = f"{sign}{v:.0f} m/s"
        txt = ax.text(
            xm, tm, label,
            color=col, fontsize=9, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(facecolor=PANEL_BG, edgecolor=col,
                      alpha=0.85, boxstyle="round,pad=0.3"),
            zorder=7,
        )

        pick["artists"] = [line, dot0, dot1, txt]

    def _remove_pick_artists(self, pick):
        for artist in pick.get("artists", []):
            try:
                artist.remove()
            except Exception:
                pass
        pick["artists"] = []

    # ----------------------------------------------------------
    # 速度计算
    # ----------------------------------------------------------

    @staticmethod
    def _calc_velocity(x0, t0, x1, t1):
        """
        视速度 = ΔX / ΔT。
        正值：向右传播；负值：向左传播。
        """
        dt = t1 - t0
        dx = x1 - x0
        if abs(dt) < 1e-6:
            return float("inf")
        return dx / dt

    # ----------------------------------------------------------
    # 事件连接
    # ----------------------------------------------------------

    def _connect_events(self):
        fig = self.fig
        fig.canvas.mpl_connect("button_press_event",   self._on_press)
        fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("key_press_event",      self._on_key)
        fig.canvas.mpl_connect("close_event",          self._on_close)

    # ----------------------------------------------------------
    # 鼠标事件
    # ----------------------------------------------------------

    def _on_press(self, event):
        if event.inaxes != self.ax_seis:
            return

        # 右键：命中检测 → 删除最近线条
        if event.button == 3:
            self._try_delete_pick(event.xdata, event.ydata)
            return

        # 左键：开始拖线
        if event.button == 1:
            self._drag_start  = (event.xdata, event.ydata)
            self._dragging    = True
            self._temp_line   = None

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax_seis:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, t0 = self._drag_start
        x1, t1 = event.xdata, event.ydata

        # 移除旧临时线
        if self._temp_line is not None:
            for a in self._temp_line:
                try:
                    a.remove()
                except Exception:
                    pass

        col = COLORS[self.color_idx % len(COLORS)]

        # 实时速度
        v = self._calc_velocity(x0, t0, x1, t1)
        v_str = f"{v:+.0f} m/s" if abs(v) < 1e5 else "∞ m/s"

        line, = self.ax_seis.plot([x0, x1], [t0, t1],
                                   color=col, lw=1.5, ls="--",
                                   alpha=0.7, zorder=5)
        txt = self.ax_seis.text(
            x1, t1,
            f" {v_str}",
            color=col, fontsize=10, fontweight="bold",
            va="center", zorder=8,
        )
        self._temp_line = [line, txt]
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False

        # 清临时线
        if self._temp_line is not None:
            for a in self._temp_line:
                try:
                    a.remove()
                except Exception:
                    pass
            self._temp_line = None

        if event.inaxes != self.ax_seis:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, t0 = self._drag_start
        x1, t1 = event.xdata, event.ydata

        # 拖动距离太短则忽略（防误触）
        px_dist = ((x1 - x0) ** 2 + (t1 - t0) ** 2) ** 0.5
        if px_dist < self.dx * 0.5:
            self.fig.canvas.draw_idle()
            return

        v   = self._calc_velocity(x0, t0, x1, t1)
        col = COLORS[self.color_idx % len(COLORS)]
        self.color_idx += 1

        pick = dict(x0=x0, t0=t0, x1=x1, t1=t1, v=v, color=col, artists=[])
        self.picks.append(pick)
        self._draw_pick_artists(pick)
        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def _try_delete_pick(self, xd, td):
        """找到距右键点击最近的线条，若在容差内则删除。"""
        if not self.picks:
            return

        # 容差：偏移距轴 5%
        x_tol = (self.x_axis[-1] - self.x_axis[0]) * 0.05

        best_idx  = None
        best_dist = float("inf")
        for i, p in enumerate(self.picks):
            # 点到线段的距离（屏幕单位近似）
            x0, t0, x1, t1 = p["x0"], p["t0"], p["x1"], p["t1"]
            # 归一化到同量纲：时间轴 × 速度比例
            dx = x1 - x0
            dt = t1 - t0
            L2 = dx**2 + (dt * 1000)**2  # dt 转 ms 量纲匹配
            if L2 < 1e-10:
                continue
            t_param = ((xd - x0) * dx + (td - t0) * dt * 1e6) / L2
            t_param = max(0.0, min(1.0, t_param))
            px = x0 + t_param * dx
            pt = t0 + t_param * dt
            dist = ((xd - px)**2 + ((td - pt) * 1000)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        if best_idx is not None and best_dist < x_tol:
            pick = self.picks.pop(best_idx)
            self._remove_pick_artists(pick)
            self._update_info_panel()
            self.fig.canvas.draw_idle()

    # ----------------------------------------------------------
    # 键盘事件
    # ----------------------------------------------------------

    def _on_key(self, event):
        key = event.key.lower() if event.key else ""

        if key == "c":
            self._clear_all()
        elif key == "s":
            self._save_picks()
        elif key == "f":
            self._toggle_fk()
        elif key == "r":
            self._reset_view()

    def _clear_all(self):
        for pick in self.picks:
            self._remove_pick_artists(pick)
        self.picks.clear()
        self.color_idx = 0
        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def _save_picks(self):
        if not self.picks:
            print("[保存] 没有拾取结果")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path.cwd() / f"velocity_picks_{ts}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# LNF 面波速度拾取结果  {ts}\n")
            f.write(f"# dt={self.dt} s  dx={self.dx} m\n")
            f.write(f"# {'#':>3}  {'X0(m)':>8}  {'T0(s)':>8}  "
                    f"{'X1(m)':>8}  {'T1(s)':>8}  {'V(m/s)':>10}  "
                    f"{'|V|(m/s)':>10}\n")
            for i, p in enumerate(self.picks):
                f.write(
                    f"  {i+1:3d}  {p['x0']:8.1f}  {p['t0']:8.4f}  "
                    f"{p['x1']:8.1f}  {p['t1']:8.4f}  "
                    f"{p['v']:10.2f}  {abs(p['v']):10.2f}\n"
                )
            vs = [abs(p["v"]) for p in self.picks]
            f.write(f"\n# 统计：最小={min(vs):.1f}  最大={max(vs):.1f}  "
                    f"均值={np.mean(vs):.1f}  中位数={np.median(vs):.1f} m/s\n")
        print(f"[保存] 已写入：{path}")

        # 在图上短暂提示
        msg = self.ax_seis.text(
            0.5, 0.02, f"✓ 已保存：{path.name}",
            transform=self.ax_seis.transAxes,
            ha="center", va="bottom",
            fontsize=10, color="#44FF88",
            bbox=dict(facecolor=PANEL_BG, alpha=0.9,
                      edgecolor="#44FF88", boxstyle="round"),
            zorder=20,
        )
        self.fig.canvas.draw_idle()
        # 3 秒后移除提示（利用定时器）
        def _remove_msg(*_):
            try:
                msg.remove()
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        timer = self.fig.canvas.new_timer(interval=3000)
        timer.single_shot = True
        timer.add_callback(_remove_msg)
        timer.start()

    def _toggle_fk(self):
        self._fk_visible = not self._fk_visible
        self.ax_fk.set_visible(self._fk_visible)
        if self._fk_visible:
            self._draw_fk()
        self.fig.canvas.draw_idle()

    def _reset_view(self):
        self.ax_seis.set_xlim(
            self.x_axis[0]  - self.dx / 2,
            self.x_axis[-1] + self.dx / 2,
        )
        self.ax_seis.set_ylim(
            self.t_axis[-1] + self.dt / 2,
            self.t_axis[0]  - self.dt / 2,
        )
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        if self.picks:
            self._save_picks()

    # ----------------------------------------------------------
    # 启动
    # ----------------------------------------------------------

    def show(self):
        plt.show()


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="交互式面波速度拾取工具"
    )
    parser.add_argument(
        "data", nargs="?",
        default=r"D:\桌面\面波压制\深层数据面波压制\最终数据\original.npy",
        help="输入 .npy 文件路径（默认：original.npy）",
    )
    parser.add_argument("--dt", type=float, default=0.00025, help="时间采样间隔 s")
    parser.add_argument("--dx", type=float, default=3.0,  help="道间距 m")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"[错误] 文件不存在：{path}")
        print("用法：python velocity_picker.py path/to/data.npy --dt 0.004 --dx 25")
        sys.exit(1)

    print(f"读取数据：{path}")
    data = load_data(path)
    nt, nx = data.shape
    print(f"数据尺寸：{nt} 采样点 × {nx} 道")
    print(f"时间范围：0–{nt * args.dt:.3f} s")
    print(f"偏移距范围：0–{(nx-1) * args.dx:.0f} m")
    print()
    print("操作说明：")
    print("  左键拖动  → 画线测速")
    print("  右键点线  → 删除该线")
    print("  C         → 清除全部")
    print("  S         → 保存结果")
    print("  F         → 显示/隐藏 F-K 频谱")
    print("  R         → 重置视图")
    print()

    picker = VelocityPicker(
        data, dt=args.dt, dx=args.dx,
        title=f"炮集速度拾取  —  {path.name}"
    )
    picker.show()


if __name__ == "__main__":
    main()