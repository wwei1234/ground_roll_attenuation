"""
面波区域手动标注工具
====================
用途：对文件夹中 000~191 共 192 个炮集 npy 文件逐一手动标注面波区域，
      生成同尺寸的 0/1 掩码数组并保存。

使用方法：
    python surface_wave_annotator.py

配置：修改脚本底部 CONFIG 区域中的路径即可。

操作说明：
    - 左键单击：在图上添加多边形顶点
    - 右键单击：闭合当前多边形（至少需要 3 个点）
    - Z 键：撤销上一个顶点
    - C 键：清空当前未完成的多边形
    - Delete 键：删除最后一个已完成的多边形
    - 点击"完成标注"：保存掩码并进入下一炮
    - 点击"无面波/跳过"：保存全 0 掩码并进入下一炮
    - 点击"退出程序"：立即保存进度并退出（下次可从断点继续）
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
from matplotlib.path import Path
from matplotlib import rcParams

# 支持中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

matplotlib.use('TkAgg')   # 如果 TkAgg 不可用，可改为 'Qt5Agg' 或 'MacOSX'


# ==============================
# 标注器类
# ==============================
class SurfaceWaveAnnotator:
    """
    单炮手动标注面波区域。
    支持多个封闭多边形，生成 0/1 掩码。
    """

    def __init__(self, gather: np.ndarray, shot_name: str):
        """
        Parameters
        ----------
        gather    : 2D numpy array，形状 (time_samples, n_traces)
        shot_name : 显示用的炮集名称（如 "Shot 005"）
        """
        self.gather = gather
        self.shot_name = shot_name
        self.n_time, self.n_trace = gather.shape

        # 已完成的多边形列表，每个元素为 list of (x, y) — x=trace, y=time
        self.polygons = []
        # 当前正在绘制的多边形顶点
        self.current_polygon = []

        # matplotlib 临时绘图对象
        self.current_line = None
        self.current_point_artists = []
        self.polygon_patches = []

        # 结果
        self.mask = None          # 最终 0/1 ndarray
        self.status = None        # 'done' | 'skip' | 'quit'

    # ---------- 掩码生成 ----------
    def _build_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_time, self.n_trace), dtype=np.uint8)
        if not self.polygons:
            return mask

        # 生成所有像素坐标网格（x=trace 方向，y=time 方向）
        xs = np.arange(self.n_trace)
        ys = np.arange(self.n_time)
        xx, yy = np.meshgrid(xs, ys)                        # (n_time, n_trace)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)    # (N, 2)

        for poly in self.polygons:
            if len(poly) < 3:
                continue
            path = Path(np.array(poly), closed=True)
            inside = path.contains_points(pts, radius=1e-9)
            mask |= inside.reshape(self.n_time, self.n_trace).astype(np.uint8)
        return mask

    # ---------- 清理临时绘图 ----------
    def _clear_temp(self):
        if self.current_line is not None:
            try:
                self.current_line.remove()
            except Exception:
                pass
            self.current_line = None
        for artist in self.current_point_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.current_point_artists = []

    # ---------- 重绘当前未完成多边形 ----------
    def _redraw_current(self):
        self._clear_temp()
        if not self.current_polygon:
            return
        pts = np.array(self.current_polygon)
        self.current_line, = self.ax.plot(pts[:, 0], pts[:, 1],
                                          'r-', linewidth=1.5, zorder=5)
        for x, y in self.current_polygon:
            pt, = self.ax.plot(x, y, 'ro', markersize=6, zorder=6)
            self.current_point_artists.append(pt)

    # ---------- 事件回调 ----------
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        if event.button == 1:           # 左键：添加顶点
            x, y = event.xdata, event.ydata
            self.current_polygon.append((x, y))
            self._redraw_current()
            self.fig.canvas.draw_idle()

        elif event.button == 3:         # 右键：闭合多边形
            if len(self.current_polygon) < 3:
                print("  [提示] 至少需要 3 个点才能闭合多边形")
                return
            self.polygons.append(self.current_polygon[:])
            # 绘制填充补丁
            patch = Polygon(np.array(self.current_polygon),
                            closed=True, fill=True, alpha=0.35,
                            facecolor='red', edgecolor='red', linewidth=1.5, zorder=4)
            self.ax.add_patch(patch)
            self.polygon_patches.append(patch)
            print(f"  [+] 完成第 {len(self.polygons)} 个面波多边形")
            self._clear_temp()
            self.current_polygon = []
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'z':            # 撤销上一个顶点
            if self.current_polygon:
                self.current_polygon.pop()
                self._redraw_current()
                self.fig.canvas.draw_idle()
                print(f"  [Z] 撤销顶点，当前剩余 {len(self.current_polygon)} 个点")

        elif event.key == 'c':          # 清空当前未完成多边形
            self.current_polygon = []
            self._clear_temp()
            self.fig.canvas.draw_idle()
            print("  [C] 已清空当前未完成多边形")

        elif event.key == 'delete':     # 删除最后一个已完成多边形
            if self.polygons:
                self.polygons.pop()
                patch = self.polygon_patches.pop()
                patch.remove()
                self.fig.canvas.draw_idle()
                print(f"  [-] 已删除最后一个多边形，当前共 {len(self.polygons)} 个")

    # ---------- 按钮回调 ----------
    def _on_finish(self, event):
        if not self.polygons:
            print("  [!] 没有标注任何多边形，生成全 0 掩码")
            return
        print(f"  [✓] 标注完成，共 {len(self.polygons)} 个多边形")
        self.mask = self._build_mask()
        self.status = 'done'
        plt.close(self.fig)

    def _on_skip(self, event):
        print("  [→] 跳过该炮（无面波），生成全 0 掩码")
        self.mask = np.zeros((self.n_time, self.n_trace), dtype=np.uint8)
        self.status = 'skip'
        plt.close(self.fig)

    def _on_quit(self, event):
        print("  [!] 用户请求退出程序")
        self.mask = None
        self.status = 'quit'
        plt.close(self.fig)

    # ---------- 主入口 ----------
    def run(self):
        """
        打开交互窗口，等待用户操作。
        返回 (mask, status)：
            mask   : np.ndarray (uint8) 或 None（退出时）
            status : 'done' | 'skip' | 'quit'
        """
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        plt.subplots_adjust(bottom=0.13, top=0.93)

        # ── 修复：禁用与自定义快捷键冲突的 matplotlib 内置键绑定 ──────
        # matplotlib 默认将 z/Z 绑定为"后退导航"，将 c 绑定为"前进导航"，
        # 会在用户回调之前拦截这些按键，导致 _on_key 收不到事件。
        for key in ('z', 'Z'):
            if key in plt.rcParams.get('keymap.back', []):
                plt.rcParams['keymap.back'].remove(key)
        for key in ('c',):
            if key in plt.rcParams.get('keymap.forward', []):
                plt.rcParams['keymap.forward'].remove(key)
        # 同时禁用其他可能干扰的内置键（按需添加）
        for key in ('delete',):
            if key in plt.rcParams.get('keymap.zoom', []):
                plt.rcParams['keymap.zoom'].remove(key)
        # ────────────────────────────────────────────────────────────────

        # 显示炮集
        vmax = np.percentile(np.abs(self.gather), 99) or 1.0
        self.ax.imshow(self.gather, aspect='auto', cmap='seismic',
                       vmin=-vmax, vmax=vmax, interpolation='bilinear',
                       extent=[0, self.n_trace - 1, self.n_time - 1, 0])
        self.ax.set_xlabel("道号 (Trace)", fontsize=11)
        self.ax.set_ylabel("时间采样点 (Time sample)", fontsize=11)
        self.ax.set_title(
            f"{self.shot_name}  |  形状: {self.n_time} × {self.n_trace}\n"
            "左键=添加点  右键=闭合多边形  Z=撤销点  C=清空  Delete=删最后多边形",
            fontsize=11, fontweight='bold'
        )

        # 按钮
        ax_finish = plt.axes([0.60, 0.03, 0.12, 0.055])
        ax_skip   = plt.axes([0.74, 0.03, 0.12, 0.055])
        ax_quit   = plt.axes([0.88, 0.03, 0.10, 0.055])

        btn_finish = Button(ax_finish, '完成标注', color='lightgreen')
        btn_skip   = Button(ax_skip,   '无面波/跳过', color='lightyellow')
        btn_quit   = Button(ax_quit,   '退出程序', color='lightsalmon')

        btn_finish.on_clicked(self._on_finish)
        btn_skip.on_clicked(self._on_skip)
        btn_quit.on_clicked(self._on_quit)

        # 事件绑定
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show()   # 阻塞，直到窗口关闭

        return self.mask, self.status


# ==============================
# 主流程
# ==============================
def run_annotation(
    input_dir: str,
    output_dir: str,
    total: int = 192,
    start_idx: int = 0,
    prefix_in: str = "",
    prefix_out: str = "mask_",
):
    """
    逐一打开炮集文件，标注后保存掩码。

    Parameters
    ----------
    input_dir  : 炮集 npy 文件所在文件夹（文件名格式：{prefix_in}000.npy ~ ...191.npy）
    output_dir : 掩码 npy 文件保存文件夹
    total      : 炮集总数（默认 192，编号 000~191）
    start_idx  : 从哪一炮开始（支持断点续标，默认 0）
    prefix_in  : 输入文件名前缀（如 "shot_"，则文件名为 shot_000.npy）
    prefix_out : 输出掩码文件名前缀（默认 "mask_"）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 检测已标注的掩码，自动跳过
    existing = set()
    for f in glob.glob(os.path.join(output_dir, f"{prefix_out}*.npy")):
        base = os.path.splitext(os.path.basename(f))[0]
        num_str = base[len(prefix_out):]
        try:
            existing.add(int(num_str))
        except ValueError:
            pass

    print("=" * 60)
    print("面波区域手动标注工具")
    print("=" * 60)
    print(f"输入目录  : {input_dir}")
    print(f"输出目录  : {output_dir}")
    print(f"总炮数    : {total}  (编号 {0:03d} ~ {total-1:03d})")
    print(f"已完成    : {len(existing)} 炮")
    if existing:
        done_list = sorted(existing)
        print(f"  已标注编号: {done_list[:10]}{'...' if len(done_list) > 10 else ''}")
    print("=" * 60)

    for idx in range(start_idx, total):
        num_str = f"{idx:03d}"
        in_path  = os.path.join(input_dir,  f"{prefix_in}{num_str}.npy")
        out_path = os.path.join(output_dir, f"{prefix_out}{num_str}.npy")

        # 跳过已标注
        if idx in existing:
            print(f"[{idx+1}/{total}] Shot {num_str} — 已有掩码，跳过")
            continue

        # 检查输入文件
        if not os.path.exists(in_path):
            print(f"[{idx+1}/{total}] Shot {num_str} — 文件不存在: {in_path}，跳过")
            continue

        # 加载炮集
        gather = np.load(in_path).astype(np.float32)
        if gather.ndim != 2:
            print(f"  [警告] 数据维度为 {gather.ndim}，期望 2D，跳过")
            continue

        print(f"\n[{idx+1}/{total}] Shot {num_str}  |  形状: {gather.shape}")

        annotator = SurfaceWaveAnnotator(gather, shot_name=f"Shot {num_str}")
        mask, status = annotator.run()

        if status == 'quit':
            print("\n[!] 程序退出，未保存当前炮的掩码。已标注的炮均已保存。")
            sys.exit(0)

        # 保存掩码
        np.save(out_path, mask)
        sw_count = int(mask.sum())
        total_px  = mask.size
        print(f"  [✓] 掩码已保存 → {out_path}")
        print(f"      面波像素: {sw_count} / {total_px}  ({100*sw_count/total_px:.2f}%)")

    print("\n" + "=" * 60)
    print("全部标注完成！")
    print(f"掩码保存于: {output_dir}")
    print("=" * 60)


# ==============================
# ★ 配置区域（按需修改）★
# ==============================
if __name__ == "__main__":

    # ---- 路径配置 ----
    INPUT_DIR  = r"原始炮集"      # 炮集 npy 所在文件夹
    OUTPUT_DIR = r"masks"     # 掩码输出文件夹

    # ---- 文件名配置 ----
    PREFIX_IN  = "real_data_gather"   # 输入文件名前缀
    PREFIX_OUT = "mask_"          # 输出掩码文件名前缀

    # ---- 范围配置 ----
    TOTAL      = 192   # 炮集总数
    START_IDX  = 183     # 从第几炮开始

    run_annotation(
        input_dir  = INPUT_DIR,
        output_dir = OUTPUT_DIR,
        total      = TOTAL,
        start_idx  = START_IDX,
        prefix_in  = PREFIX_IN,
        prefix_out = PREFIX_OUT,
    )