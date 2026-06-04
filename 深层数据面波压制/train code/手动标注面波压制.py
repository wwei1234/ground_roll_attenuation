"""
交互式面波区域标注 + 面波压制推理脚本
=========================================
使用方法：
  1. 运行脚本，弹出炮集图像窗口
  2. 在图上点击多个点，画出面波区域的多边形边界
  3. 一个多边形画完后按 Enter 键确认，可继续画下一个多边形
  4. 所有多边形画完后按 'd' 键完成标注，开始推理
  5. 按 'z' 键撤销上一个多边形
  6. 按 'c' 键清除所有多边形重新画

快捷键汇总：
  鼠标左键  - 添加多边形顶点
  Enter     - 完成当前多边形（闭合并开始下一个）
  z         - 撤销上一个已完成的多边形
  c         - 清除所有多边形
  d         - 完成所有标注，开始推理
  q / 关闭  - 退出
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import rcParams
from U_Net_CBAM import UNet


# ============================================================
#  ★ 配置参数（修改这里）
# ============================================================
CONFIG = {
    # ---------- 路径 ----------
    "model_path"  : r"CU-Net_3_5\best_model.pth",
    "gather_path" : r"D:\桌面\面波压制\LNF\LNF_output\LNF_denoised.npy",
    "output_dir"  : r"面波压制结果\new_manual_mask_results",

    # ---------- 预处理 ---------
    "normalize"        : True,
    "max_time_samples" : None,

    # ---------- 高斯噪声（与训练时 mask_k 一致）----------
    "fixed_mask_value" : 0.5,

    # ---------- 后处理 ----------
    "replace_non_sw_with_input" : True,
    "boost_sw_energy"           : True,
    "energy_boost_factor"       : 1,

    # ---------- 设备 ----------
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
}


# ============================================================
#  工具函数
# ============================================================

def normalize_traces_per_trace(shot_gather: np.ndarray) -> np.ndarray:
    normalized = shot_gather.copy()
    for i in range(normalized.shape[1]):
        trace = normalized[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized[:, i] = trace / max_amp
    return normalized


def polygons_to_mask(polygons_xy: list, shape: tuple) -> np.ndarray:
    """
    将多个多边形（以图像坐标 x=道号, y=时间采样点 表示）转换为二值掩码。
    shape: (n_time, n_trace)
    polygons_xy: list of np.ndarray, 每个形状 (N, 2)，列顺序 [x, y]
    """
    n_time, n_trace = shape
    mask = np.zeros((n_time, n_trace), dtype=np.uint8)

    # 构建所有像素的坐标网格
    x_coords = np.arange(n_trace)
    y_coords = np.arange(n_time)
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])  # (N*M, 2) -> [x, y]

    for poly_xy in polygons_xy:
        if len(poly_xy) < 3:
            continue
        path = Path(poly_xy)
        inside = path.contains_points(points)
        mask_flat = inside.reshape(n_time, n_trace)
        mask = np.logical_or(mask, mask_flat).astype(np.uint8)

    return mask


# ============================================================
#  交互式标注器
# ============================================================

class InteractiveAnnotator:
    """在炮集图像上交互式画多边形，生成面波掩码。"""

    def __init__(self, gather: np.ndarray):
        self.gather      = gather
        self.n_time, self.n_trace = gather.shape

        self.finished_polygons = []   # list of np.ndarray (N,2) [x,y]
        self.current_points    = []   # 当前正在画的多边形顶点
        self.done              = False

        self._setup_figure()

    def _setup_figure(self):
        vmax = np.percentile(np.abs(self.gather), 99) or 1.0
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(
            self.gather, aspect='auto', cmap='seismic',
            vmin=-vmax, vmax=vmax, interpolation='bilinear'
        )
        self.ax.set_title(
            '面波区域标注\n'
            '左键点击添加顶点 | Enter 完成当前多边形 | '
            'd 完成标注开始推理 | z 撤销 | c 清除',
            fontsize=10
        )
        self.ax.set_xlabel('道号')
        self.ax.set_ylabel('时间采样点')

        # 用于实时显示当前点和已完成多边形的绘图对象
        self.current_line, = self.ax.plot([], [], 'y-o', linewidth=1.5,
                                          markersize=5, label='当前多边形')
        self.status_text = self.ax.text(
            0.01, 0.99, '多边形数量: 0  |  当前顶点: 0',
            transform=self.ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6),
            color='white'
        )

        self.patch_artists = []  # 已完成多边形的填充色块

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',    self._on_key)

    def _update_current_line(self):
        if len(self.current_points) == 0:
            self.current_line.set_data([], [])
        else:
            pts = np.array(self.current_points)
            xs = list(pts[:, 0]) + [pts[0, 0]]  # 闭合预览
            ys = list(pts[:, 1]) + [pts[0, 1]]
            self.current_line.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def _update_status(self):
        self.status_text.set_text(
            f'多边形数量: {len(self.finished_polygons)}  |  '
            f'当前顶点: {len(self.current_points)}'
        )
        self.fig.canvas.draw_idle()

    def _draw_finished_polygon(self, poly_xy: np.ndarray, color: str):
        patch = Polygon(poly_xy, closed=True,
                        facecolor=color, edgecolor='yellow',
                        alpha=0.35, linewidth=1.5)
        self.ax.add_patch(patch)
        self.patch_artists.append(patch)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        self.current_points.append([x, y])
        self._update_current_line()
        self._update_status()

    def _on_key(self, event):
        key = event.key

        # Enter：完成当前多边形
        if key == 'enter':
            if len(self.current_points) >= 3:
                poly = np.array(self.current_points)
                self.finished_polygons.append(poly)
                colors = ['red', 'cyan', 'magenta', 'orange', 'lime', 'violet']
                color  = colors[len(self.finished_polygons) % len(colors)]
                self._draw_finished_polygon(poly, color)
                self.current_points = []
                self._update_current_line()
                self._update_status()
                print(f"  [✓] 多边形 {len(self.finished_polygons)} 已添加"
                      f"（{len(poly)} 个顶点）")
            else:
                print("  [!] 至少需要 3 个顶点才能构成多边形")

        # z：撤销上一个已完成的多边形
        elif key == 'z':
            if self.finished_polygons:
                self.finished_polygons.pop()
                if self.patch_artists:
                    self.patch_artists[-1].remove()
                    self.patch_artists.pop()
                self._update_status()
                self.fig.canvas.draw_idle()
                print(f"  [z] 已撤销，当前多边形数量: {len(self.finished_polygons)}")
            else:
                # 撤销当前点的最后一个顶点
                if self.current_points:
                    self.current_points.pop()
                    self._update_current_line()

        # c：清除所有
        elif key == 'c':
            self.finished_polygons.clear()
            self.current_points.clear()
            for p in self.patch_artists:
                p.remove()
            self.patch_artists.clear()
            self._update_current_line()
            self._update_status()
            self.fig.canvas.draw_idle()
            print("  [c] 已清除所有多边形")

        # d：完成标注
        elif key == 'd':
            # 如果当前还有未完成的多边形，自动闭合
            if len(self.current_points) >= 3:
                poly = np.array(self.current_points)
                self.finished_polygons.append(poly)
                colors = ['red', 'cyan', 'magenta', 'orange', 'lime', 'violet']
                color  = colors[len(self.finished_polygons) % len(colors)]
                self._draw_finished_polygon(poly, color)
                self.current_points = []
                self._update_current_line()
                print(f"  [✓] 自动闭合最后一个多边形（{len(poly)} 个顶点）")

            if len(self.finished_polygons) == 0:
                print("  [!] 请至少画一个多边形再按 d")
                return

            self.done = True
            plt.title('标注完成，正在生成掩码并推理...', fontsize=11, color='lime')
            self.fig.canvas.draw_idle()
            plt.close(self.fig)

        # q：退出
        elif key == 'q':
            plt.close(self.fig)

    def run(self) -> np.ndarray:
        """显示标注窗口，返回生成的掩码 (n_time, n_trace) uint8"""
        plt.show()

        if not self.done or len(self.finished_polygons) == 0:
            print("未完成标注，退出。")
            return None

        print(f"\n共标注 {len(self.finished_polygons)} 个多边形，正在生成掩码...")
        mask = polygons_to_mask(
            self.finished_polygons,
            (self.n_time, self.n_trace)
        )
        print(f"掩码生成完成，面波占比: {mask.mean()*100:.1f}%")
        return mask


# ============================================================
#  推理器
# ============================================================

class ManualMaskDenoiser:

    def __init__(self, model_path: str, device: str = None,
                 fixed_mask_value: float = 0.5):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fixed_mask_value = fixed_mask_value

        self.model = UNet().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"加载 checkpoint（epoch={ckpt.get('epoch','?')}，"
                  f"val_loss={ckpt.get('val_loss',0):.6f}）")
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"模型已加载至 {self.device}")

    def preprocess(self, shot_gather: np.ndarray,
                   normalize: bool = True,
                   max_time_samples: int = None) -> np.ndarray:
        data = shot_gather.copy().astype(np.float32)
        if max_time_samples is not None:
            data = data[:max_time_samples, :]
        if normalize:
            data = normalize_traces_per_trace(data)
        return data

    def _build_network_input(self, gather: np.ndarray,
                             sw_mask: np.ndarray) -> np.ndarray:
        """面波区域替换为高斯噪声，与训练时一致。"""
        gaussian = np.random.normal(
            0, self.fixed_mask_value / 3.0, size=gather.shape
        ).astype(np.float32)
        gaussian = np.clip(gaussian, -self.fixed_mask_value, self.fixed_mask_value)

        network_input = gather.copy()
        network_input[sw_mask.astype(bool)] = gaussian[sw_mask.astype(bool)]
        return network_input

    @torch.no_grad()
    def denoise(self, gather: np.ndarray, sw_mask: np.ndarray,
                replace_non_sw_with_input: bool = True,
                boost_sw_energy: bool = False,
                energy_boost_factor: float = 1.0) -> dict:

        sw_bool      = sw_mask.astype(bool)
        network_input = self._build_network_input(gather, sw_mask)

        inp_tensor = torch.from_numpy(network_input).float()
        inp_tensor = inp_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        out_tensor = self.model(inp_tensor)
        network_output = out_tensor.squeeze().cpu().numpy()

        denoised = network_output.copy()
        if replace_non_sw_with_input:
            denoised[~sw_bool] = gather[~sw_bool]
        if boost_sw_energy and sw_bool.any():
            denoised[sw_bool] *= energy_boost_factor

        return {
            'preprocessed'  : gather,
            'network_input' : network_input,
            'network_output': network_output,
            'denoised'      : denoised,
            'sw_mask'       : sw_mask,
            'difference'    : gather - denoised,
        }

    def plot_results(self, results: dict, save_path: str = None,
                     figsize=(20, 10)):
        preprocessed   = results['preprocessed']
        network_input  = results['network_input']
        network_output = results['network_output']
        denoised       = results['denoised']
        sw_mask        = results['sw_mask']
        difference     = results['difference']

        vmax      = np.percentile(np.abs(preprocessed), 99) or 1.0
        vmax_diff = np.percentile(np.abs(difference),   99) or 1.0

        kw_seis = dict(aspect='auto', cmap='seismic',
                       vmin=-vmax, vmax=vmax, interpolation='bilinear')
        kw_mask = dict(aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=1, interpolation='nearest')

        panels = [
            ('1. 预处理后原始数据',   kw_seis, preprocessed),
            ('2. 手动标注面波掩码',   kw_mask, sw_mask.astype(float)),
            ('3. 网络输入',           kw_seis, network_input),
            ('4. 网络原始输出',       kw_seis, network_output),
            ('5. 最终去噪结果',       kw_seis, denoised),
            ('6. 差异（原始 - 结果）',
             dict(aspect='auto', cmap='seismic',
                  vmin=-vmax_diff, vmax=vmax_diff, interpolation='bilinear'),
             difference),
        ]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        for ax, (title, kw, data) in zip(axes.flatten(), panels):
            im = ax.imshow(data, **kw)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('道号')
            ax.set_ylabel('时间采样点')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        sw_ratio     = sw_mask.mean() * 100
        signal_ratio = (
            np.mean(np.abs(denoised[~sw_mask.astype(bool)])) /
            (np.mean(np.abs(preprocessed[~sw_mask.astype(bool)])) + 1e-10)
        ) * 100
        fig.suptitle(
            f'面波压制结果  |  面波占比: {sw_ratio:.1f}%  |  '
            f'非面波区域能量保留: {signal_ratio:.1f}%',
            fontsize=12, y=1.01
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"对比图已保存: {save_path}")
        else:
            plt.show()

    def save_results(self, results: dict, output_dir: str, prefix: str = "shot"):
        os.makedirs(output_dir, exist_ok=True)
        save_map = {
            'preprocessed'  : f"{prefix}_preprocessed.npy",
            'network_input' : f"{prefix}_network_input.npy",
            'network_output': f"{prefix}_network_output.npy",
            'denoised'      : f"{prefix}_denoised.npy",
            'sw_mask'       : f"{prefix}_sw_mask.npy",
            'difference'    : f"{prefix}_difference.npy",
        }
        for key, fname in save_map.items():
            np.save(os.path.join(output_dir, fname), results[key])
        print(f"所有结果已保存至: {output_dir}")


# ============================================================
#  主流程
# ============================================================

def main():
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

    cfg = CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # ---------- 加载炮集 ----------
    print(f"\n加载炮集: {cfg['gather_path']}")
    gather_raw = np.load(cfg['gather_path']).astype(np.float32)
    print(f"  原始形状: {gather_raw.shape}")

    # ---------- 初始化推理器 ----------
    denoiser = ManualMaskDenoiser(
        model_path       = cfg['model_path'],
        device           = cfg['device'],
        fixed_mask_value = cfg['fixed_mask_value'],
    )

    # ---------- 预处理 ----------
    gather = denoiser.preprocess(
        gather_raw,
        normalize        = cfg['normalize'],
        max_time_samples = cfg['max_time_samples'],
    )
    print(f"预处理后形状: {gather.shape}")

    # ---------- 交互式标注 ----------
    print("\n" + "=" * 55)
    print("请在弹出的窗口中标注面波区域：")
    print("  鼠标左键  → 添加多边形顶点")
    print("  Enter     → 完成当前多边形")
    print("  d         → 完成所有标注，开始推理")
    print("  z         → 撤销上一个多边形（或最后一个顶点）")
    print("  c         → 清除所有重新画")
    print("=" * 55 + "\n")

    annotator = InteractiveAnnotator(gather)
    sw_mask   = annotator.run()

    if sw_mask is None:
        print("未完成标注，程序退出。")
        return

    # ---------- 保存 mask ----------
    gather_name = os.path.splitext(os.path.basename(cfg['gather_path']))[0]
    mask_save_path = os.path.join(cfg['output_dir'], f"{gather_name}_sw_mask.npy")
    np.save(mask_save_path, sw_mask)
    print(f"面波掩码已保存: {mask_save_path}")

    # ---------- 推理 ----------
    print("\n开始推理...")
    results = denoiser.denoise(
        gather,
        sw_mask,
        replace_non_sw_with_input = cfg['replace_non_sw_with_input'],
        boost_sw_energy           = cfg['boost_sw_energy'],
        energy_boost_factor       = cfg['energy_boost_factor'],
    )
    print("推理完成")

    # ---------- 保存 & 可视化 ----------
    denoiser.save_results(results, cfg['output_dir'], prefix=gather_name)

    fig_path = os.path.join(cfg['output_dir'], f"{gather_name}_comparison.png")
    denoiser.plot_results(results, save_path=fig_path)

    print(f"\n✓ 全部完成，结果保存在: {cfg['output_dir']}")


if __name__ == "__main__":
    main()