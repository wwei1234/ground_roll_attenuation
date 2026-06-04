"""
方案二：Ra 场形态学连续化掩码生成器
=====================================

原理
----
Ra 场逐点跳变的根本原因是速度扫描没有空间约束。
本方案不放弃 Ra 场的物理意义，而是把 Ra 场当作图像，
用形态学方法提取空间连续的面波区域：

  1. 粗阈值二值化 Ra 场
  2. 形态学开运算：去掉孤立碎斑（误判的面波点）
  3. 形态学闭运算：填充面波区域内的空洞（漏判的面波点）
  4. 连通域分析：过滤面积过小的孤立区域
  5. 距离变换：生成连续软边界（替代直接用 Ra 值做软掩码）
  6. 应用到原始数据 → DMSSL 网络输入

优点：保留 Ra 场的物理意义（速度 + 频率 + 方向联合判断）
      掩码空间连续，无碎斑，软边界自然过渡
缺点：需要先跑 Ra 场计算（约 LNF 40% 的计算量）

依赖：numpy, scipy, numba, matplotlib
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import (
    binary_opening, binary_closing, binary_fill_holes,
    distance_transform_edt, label, gaussian_filter
)
from numba import njit
from scipy.ndimage import convolve1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# ===========================================================================
# Ra 场计算（复用自 ra_mask_generator.py，独立版本）
# ===========================================================================

@njit(cache=True)
def _extract_window_nb(s_pad, i0, j0, vk, dt, dx, N, M):
    half_M = M // 2
    w = np.zeros((N, M))
    nr, nc = s_pad.shape
    for j in range(M):
        shift = int(round((j - half_M) * dx / (vk * dt)))
        for i in range(N):
            r = i0 + i + shift
            c = j0 + j
            if 0 <= r < nr and 0 <= c < nc:
                w[i, j] = s_pad[r, c]
    return w


@njit(cache=True)
def _similarity_nb(w):
    N, M = w.shape
    Aj = np.zeros(M)
    for j in range(M):
        s2 = 0.0
        for i in range(N):
            s2 += w[i, j] ** 2
        v = (s2 / N) ** 0.5
        Aj[j] = v if v > 1e-10 else 1e-10
    num, den = 0.0, 0.0
    for i in range(N):
        rs = 0.0
        for j in range(M):
            val = w[i, j] / Aj[j]
            rs  += val
            den += val * val
        num += rs * rs
    den *= M
    if den < 1e-10:
        return 0.0
    return min(num / den, 1.0)


@njit(cache=True)
def _compute_ra_field_nb(s_pad, v_list, dt, dx, N, M, nt, nx):
    Nv = len(v_list)
    Ra_field   = np.zeros((nt, nx))
    Vopt_field = np.zeros((nt, nx))
    for j0 in range(nx):
        for i0 in range(nt):
            Ra_max, v_opt = 0.0, v_list[0]
            for k in range(Nv):
                w  = _extract_window_nb(s_pad, i0, j0, v_list[k], dt, dx, N, M)
                Ra = _similarity_nb(w)
                if Ra > Ra_max:
                    Ra_max = Ra
                    v_opt  = v_list[k]
            Ra_field[i0, j0]   = Ra_max
            Vopt_field[i0, j0] = v_opt
    return Ra_field, Vopt_field


def _compute_ra(data, dt, dx, f1, f2, v_min, v_max, dv, win_t, win_x):
    """频率分离 + 速度扫描，返回 Ra 场。"""
    nt, nx = data.shape
    N = int(win_t / dt)
    if N % 2 == 0: N += 1
    M = win_x if win_x % 2 == 1 else win_x + 1
    Nv     = max(1, int((v_max - v_min) / dv))
    v_list = np.linspace(v_min, v_max, Nv).astype(np.float64)

    # 低通滤波
    flen = min(101, nt // 2 * 2 + 1)
    t    = np.arange(flen) * dt - (flen // 2) * dt + 1e-10
    df   = max(f2 - f1, 1e-6)
    h    = (4*np.sin(np.pi*(f2+f1)*t)/(np.pi*t)) * \
           (np.sin(np.pi*df*t/2)**2 / (df**2*np.pi**2*t**2))
    h   /= np.sum(np.abs(h))
    s_low = convolve1d(data.astype(np.float64), h, axis=0, mode="mirror")

    # 镜像填充
    hN, hM = N//2, M//2
    s_pad  = np.pad(s_low, ((hN,hN),(hM,hM)), mode="reflect")

    # Numba 预热
    dummy = np.zeros((N*3, M*3))
    _compute_ra_field_nb(dummy, v_list[:min(3,Nv)], dt, dx, N, M, N, M)

    Ra_field, Vopt_field = _compute_ra_field_nb(
        s_pad, v_list, dt, dx, N, M, nt, nx
    )
    return Ra_field, Vopt_field


# ===========================================================================
# 核心：形态学连续化掩码
# ===========================================================================

class MorphMaskGenerator:
    """
    Ra 场形态学连续化掩码生成器。

    Parameters
    ----------
    ra_threshold : float
        粗阈值，Ra > ra_threshold 的点初始化为面波候选。
        建议设为 Ra 直方图两峰之间谷底的值。
        若不确定，先用 0.3–0.4 试。
    morph_t : int
        形态学结构元素的时间轴半径（采样点）。
        开/闭运算在时间轴方向的作用范围。建议 2–4。
    morph_x : int
        形态学结构元素的空间轴半径（道数）。
        建议等于 win_x // 2，与速度扫描窗口匹配。
    min_area : int
        连通域面积下限（采样点数）。
        小于此面积的连通域被视为碎斑并删除。
        建议设为 nx × 3（至少跨越 3 个时间采样点的宽度）。
    transition_width_t : int
        距离变换软边界的时间轴宽度（采样点）。
        控制掩码边界在时间轴方向的渐变速度。
    transition_width_x : int
        距离变换软边界的空间轴宽度（道数）。
    gamma : float
        软边界的非线性幂次（同方案一的 gamma）。
        >1 更保守，<1 更激进。
    noise_ratio : float
        高斯噪声强度 = 数据 RMS × noise_ratio。
    """

    def __init__(
        self,
        ra_threshold=0.35,
        morph_t=3,
        morph_x=5,
        min_area=None,       # None = 自动设为 nx × 3
        transition_width_t=8,
        transition_width_x=4,
        gamma=1.2,
        noise_ratio=0.1,
    ):
        self.ra_threshold       = ra_threshold
        self.morph_t            = morph_t
        self.morph_x            = morph_x
        self.min_area           = min_area
        self.transition_width_t = transition_width_t
        self.transition_width_x = transition_width_x
        self.gamma              = gamma
        self.noise_ratio        = noise_ratio

    # ------------------------------------------------------------------
    # 步骤 1：粗阈值二值化
    # ------------------------------------------------------------------

    def _binarize(self, Ra_field):
        """Ra > ra_threshold → 面波候选（True）。"""
        return Ra_field > self.ra_threshold

    # ------------------------------------------------------------------
    # 步骤 2-3：形态学开/闭运算
    # ------------------------------------------------------------------

    def _morphology(self, binary, nx):
        """
        开运算去碎斑，闭运算填空洞。

        结构元素使用矩形（morph_t × morph_x），
        形状与面波在炮集上的形态匹配：
        时间轴窄（面波波形持续时间短），空间轴宽（多道连续）。
        """
        struct = np.ones((self.morph_t * 2 + 1,
                          self.morph_x * 2 + 1), dtype=bool)

        # 开运算：先腐蚀再膨胀，去掉比结构元素小的孤立碎斑
        binary = binary_opening(binary, structure=struct, iterations=1)
        # 闭运算：先膨胀再腐蚀，填充面波区域内的孔洞
        binary = binary_closing(binary, structure=struct, iterations=1)
        # 填充完全封闭的内部空洞
        binary = binary_fill_holes(binary)

        return binary

    # ------------------------------------------------------------------
    # 步骤 4：连通域分析，过滤小面积碎斑
    # ------------------------------------------------------------------

    def _filter_components(self, binary, nx):
        """
        保留面积大于 min_area 的连通域。

        面波在炮集上是跨越多道的大块区域；
        孤立的小连通域（面积小）基本都是误判的高 Ra 点。
        """
        min_area = self.min_area if self.min_area is not None else nx * 3

        labeled, n = label(binary)
        result = np.zeros_like(binary)
        for i in range(1, n + 1):
            component = (labeled == i)
            if component.sum() >= min_area:
                result |= component

        n_removed = n - int(result.any())
        print(f"    连通域分析：共 {n} 个区域，"
              f"保留 {np.unique(labeled[result]).size - 1} 个，"
              f"删除 {n - (np.unique(labeled[result]).size - 1)} 个小碎斑")
        return result

    # ------------------------------------------------------------------
    # 步骤 5：距离变换 → 连续软边界
    # ------------------------------------------------------------------

    def _distance_to_alpha(self, binary):
        """
        用距离变换生成软掩码强度。

        与直接用 Ra 值做软掩码的区别：
        - Ra 值：逐点独立，空间不连续
        - 距离变换：基于连续的二值掩码区域，
          每个点的软强度由它到面波区域边界的距离决定，
          天然是空间连续且结构感知的。

        内部点距离大 → alpha 接近 1（完全替换）
        边界点距离小 → alpha 接近 0（保留原始数据）
        外部点距离 = 0 → alpha = 0

        过渡带宽度由 transition_width_t 和 transition_width_x 控制，
        使用各向异性距离变换（时间轴和空间轴权重不同）。
        """
        # 各向异性距离变换：时间轴和空间轴用不同的采样间距
        # 这样过渡带在两个方向上有不同的宽度
        sampling = (1.0 / self.transition_width_t,
                    1.0 / self.transition_width_x)

        dist_inside = distance_transform_edt(binary,   sampling=sampling)
        # 归一化：距离 >= 1 时 alpha = 1
        alpha = np.clip(dist_inside, 0.0, 1.0)
        # 非线性映射
        alpha = alpha ** self.gamma

        return alpha

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def generate_mask(self, Ra_field):
        """
        从 Ra 场生成形态学连续软掩码。

        Parameters
        ----------
        Ra_field : ndarray, shape (nt, nx)

        Returns
        -------
        alpha : ndarray, shape (nt, nx)
            软掩码强度 [0, 1]。
        binary_final : ndarray, shape (nt, nx)
            形态学处理后的二值掩码（调试用）。
        binary_raw : ndarray
            粗阈值二值化结果（对比用）。
        """
        nt, nx = Ra_field.shape

        print(f"  步骤 1：粗阈值二值化（threshold={self.ra_threshold}）...")
        binary_raw = self._binarize(Ra_field)
        coverage_raw = binary_raw.mean() * 100
        print(f"    初始覆盖率：{coverage_raw:.1f}%")

        print(f"  步骤 2-3：形态学开/闭运算"
              f"（结构元素 {self.morph_t*2+1}×{self.morph_x*2+1}）...")
        binary_morph = self._morphology(binary_raw, nx)

        print(f"  步骤 4：连通域分析...")
        binary_final = self._filter_components(binary_morph, nx)
        coverage_final = binary_final.mean() * 100
        print(f"    最终覆盖率：{coverage_final:.1f}%")

        print(f"  步骤 5：距离变换 → 软边界...")
        alpha = self._distance_to_alpha(binary_final)

        return alpha, binary_final, binary_raw

    def apply_mask(self, data, Ra_field, seed=None):
        """
        生成掩码并应用到数据。

        Parameters
        ----------
        data : ndarray, shape (nt, nx)
        Ra_field : ndarray, shape (nt, nx)
        seed : int or None

        Returns
        -------
        masked_data : ndarray   DMSSL 网络输入
        alpha : ndarray         软掩码强度
        binary_final : ndarray  最终二值掩码
        binary_raw : ndarray    粗阈值二值掩码
        """
        if seed is not None:
            np.random.seed(seed)

        alpha, binary_final, binary_raw = self.generate_mask(Ra_field)

        std   = np.sqrt(np.mean(data ** 2)) * self.noise_ratio
        noise = np.random.randn(*data.shape) * std

        masked_data = data * (1.0 - alpha) + noise * alpha
        return masked_data, alpha, binary_final, binary_raw

    def weighted_loss_weights(self, Ra_field):
        """生成损失函数权重（面波区 / 有效信号区）。"""
        alpha, _, _ = self.generate_mask(Ra_field)
        return alpha, 1.0 - alpha


# ===========================================================================
# 可视化
# ===========================================================================

def plot_morph_mask_overview(
    data, Ra_field, binary_raw, binary_final, alpha, masked_data,
    dt=0.004, dx=25, save_path=None
):
    """六图总览：Ra场 | 粗二值 | 形态学后 | 软掩码α | 原始 | 掩码后"""
    fig = plt.figure(figsize=(22, 8), facecolor="#0D1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.05, right=0.97,
                            top=0.93, bottom=0.08,
                            wspace=0.28, hspace=0.38)
    plt.rcParams.update({
        "text.color":"#E6EDF3","axes.labelcolor":"#E6EDF3",
        "xtick.color":"#E6EDF3","ytick.color":"#E6EDF3",
        "axes.edgecolor":"#30363D","axes.facecolor":"#161B22",
    })

    nt, nx = data.shape
    ext = [0, nx*dx, nt*dt, 0]
    vmax = np.percentile(np.abs(data), 95)

    def axi(pos, d, title, cmap, vmin, vmax, clabel=""):
        ax = fig.add_subplot(gs[pos])
        im = ax.imshow(d, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=ext)
        ax.set_title(title, fontsize=10, color="#58A6FF", pad=5)
        ax.set_xlabel("offset (m)", fontsize=8)
        ax.set_ylabel("time (s)",   fontsize=8)
        cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(clabel, fontsize=7, color="#E6EDF3")
        cb.ax.tick_params(colors="#E6EDF3", labelsize=7)

    axi((0,0), Ra_field,     "Ra field",            "hot",    0,    1,  "Ra")
    axi((0,1), binary_raw.astype(float),
                              "Binary (raw threshold)",
                                                     "Blues",  0,    1,  "")
    axi((0,2), binary_final.astype(float),
                              "Binary (after morphology)",
                                                     "Blues",  0,    1,  "")
    axi((1,0), alpha,         "Soft mask alpha",     "Blues",  0,    1,  "alpha")
    axi((1,1), data,          "Original shot",       "seismic",-vmax, vmax,"amp")
    axi((1,2), masked_data,   "DMSSL input (masked)","seismic",-vmax, vmax,"amp")

    fig.suptitle("Method 2: Morphological Continuous Mask from Ra Field",
                 fontsize=13, color="#58A6FF", y=0.99)
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    return fig


def plot_two_method_comparison(alpha_fk, alpha_morph,
                                dt=0.004, dx=25,
                                nt=None, nx=None, save_path=None):
    """两种方案掩码对比：F-K 域 vs 形态学。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0D1117")
    plt.rcParams.update({
        "text.color":"#E6EDF3","axes.labelcolor":"#E6EDF3",
        "xtick.color":"#E6EDF3","ytick.color":"#E6EDF3",
        "axes.edgecolor":"#30363D","axes.facecolor":"#161B22",
    })
    if nt is None: nt = alpha_fk.shape[0]
    if nx is None: nx = alpha_fk.shape[1]
    ext = [0, nx*dx, nt*dt, 0]

    items = [
        (alpha_fk,           "Method 1: F-K Domain Mask",        "Oranges"),
        (alpha_morph,        "Method 2: Morphological Mask",      "Blues"),
        (alpha_fk-alpha_morph, "Difference (FK - Morph)",         "RdBu_r"),
    ]
    for ax, (d, title, cmap) in zip(axes, items):
        vabs = max(np.abs(d).max(), 1e-6)
        vm   = 1.0 if cmap != "RdBu_r" else vabs
        vi   = 0.0 if cmap != "RdBu_r" else -vabs
        im   = ax.imshow(d, aspect="auto", cmap=cmap,
                         vmin=vi, vmax=vm, extent=ext)
        ax.set_title(title, fontsize=11, color="#58A6FF", pad=6)
        ax.set_xlabel("offset (m)", fontsize=9)
        ax.set_ylabel("time (s)",   fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle("Mask Comparison: FK Domain vs Morphological",
                 fontsize=13, color="#58A6FF")
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    return fig


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == "__main__":

    input_path = Path(r"D:\桌面\面波压制\深层数据面波压制\最终数据\original.npy")
    output_dir = Path.cwd() / "mask_morph_output"
    output_dir.mkdir(exist_ok=True)

    dt, dx = 0.004, 25

    # Ra 场计算参数
    ra_params = dict(
        f1=0, f2=25,
        v_min=100, v_max=1200, dv=25,
        win_t=0.05, win_x=7,
    )

    # 形态学掩码参数
    # ra_threshold  : 先看 Ra 直方图再定，建议两峰之间的谷底值
    # morph_t/morph_x : 结构元素尺寸，与面波形态匹配
    # transition_width : 软边界宽度，越大过渡越缓
    morph_params = dict(
        ra_threshold       = 0.35,
        morph_t            = 3,
        morph_x            = 5,
        min_area           = None,   # 自动设为 nx × 3
        transition_width_t = 8,
        transition_width_x = 4,
        gamma              = 1.2,
        noise_ratio        = 0.1,
    )

    print("=" * 55)
    print("方案二：Ra 场形态学连续化掩码生成")
    print("=" * 55)

    data = np.load(str(input_path)).astype(float).squeeze()
    assert data.ndim == 2
    nt, nx = data.shape
    print(f"数据尺寸：{nt} x {nx}")

    # 计算 Ra 场（若已有 Ra_field.npy 可直接加载跳过此步）
    ra_cache = output_dir / "Ra_field.npy"
    if ra_cache.exists():
        print(f"\n加载缓存 Ra 场：{ra_cache}")
        Ra_field = np.load(str(ra_cache))
    else:
        print("\n计算 Ra 场（首次运行需要时间）...")
        Ra_field, _ = _compute_ra(data, dt, dx, **ra_params)
        np.save(str(ra_cache), Ra_field)
        print(f"Ra 场已缓存：{ra_cache}")

    print(f"\nRa 统计：min={Ra_field.min():.3f}  "
          f"max={Ra_field.max():.3f}  mean={Ra_field.mean():.3f}")

    # 打印 Ra 直方图分位数，辅助设定 ra_threshold
    print("\nRa 分位数（辅助设定 ra_threshold）：")
    for q in [10, 25, 50, 60, 70, 75, 80, 90]:
        print(f"  {q:3d}%: {np.percentile(Ra_field, q):.3f}")

    print("\n生成形态学掩码...")
    gen = MorphMaskGenerator(**morph_params)
    masked_data, alpha, binary_final, binary_raw = gen.apply_mask(
        data, Ra_field, seed=42
    )

    np.save(str(output_dir / "morph_alpha.npy"),       alpha)
    np.save(str(output_dir / "morph_masked_data.npy"), masked_data)

    print(f"\n最终掩码覆盖率：{alpha.mean()*100:.1f}%")
    print(f"软边界比例    ：{((alpha>0.05)&(alpha<0.95)).mean()*100:.1f}%")

    print("\n生成可视化...")
    plot_morph_mask_overview(
        data, Ra_field, binary_raw, binary_final, alpha, masked_data,
        dt=dt, dx=dx,
        save_path=output_dir / "morph_mask_overview.png"
    )
    print(f"\n完成，结果保存至：{output_dir}")
    print("\n调参建议：")
    print("  1. 先看 Ra 分位数，把 ra_threshold 设在两峰之间的谷底")
    print("  2. morph_x 建议等于 win_x // 2（与速度扫描窗口匹配）")
    print("  3. transition_width_t/x 控制软边界宽度，越大掩码边界越平滑")
    print("  4. min_area 建议 nx×3，面波区域通常跨越整个炮集宽度")