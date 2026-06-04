"""
方案一：F-K 域自动掩码生成器
==============================

原理
----
面波在 F-K 域表现为以原点为顶点、沿视速度方向展开的楔形能量区。
这个楔形在 F-K 域天然是空间连续的，避免了逐点速度扫描的跳变问题。

流程
----
1. 数据 → 二维 FFT → F-K 振幅谱
2. 在 F-K 域按视速度范围划出楔形掩码区域
3. 楔形边界做软过渡（避免吉布斯效应）
4. F-K 域掩码 → 逆 FFT → 时空域软掩码 α(t, x)
5. α(t, x) 应用到原始数据 → DMSSL 网络输入

优点：掩码空间连续性最好，无碎斑
缺点：全局线性假设，不规则面波（双曲线面波）效果打折扣

依赖：numpy, scipy, matplotlib
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# ===========================================================================
# 核心：F-K 域掩码生成
# ===========================================================================

class FKMaskGenerator:
    """
    F-K 域面波软掩码生成器。

    Parameters
    ----------
    dt : float
        时间采样间隔 (s)。
    dx : float
        道间距 (m)。
    v_min : float
        面波最小视速度 (m/s)，掩码楔形的右边界斜率。
    v_max : float
        面波最大视速度 (m/s)，掩码楔形的左边界斜率。
        设为 None 表示不限制最大速度（楔形延伸到 k=0 轴）。
    f_max_noise : float
        面波最大频率 (Hz)，掩码楔形的频率上限。
        应设为低通滤波器截止频率，如 25 Hz。
    f_min_noise : float
        面波最小频率 (Hz)，通常设 0。
    taper_v : float
        速度边界的软过渡带宽，单位 m/s。
        控制楔形左右边界的渐变宽度，避免截断伪影。
    taper_f : float
        频率边界的软过渡带宽，单位 Hz。
        控制楔形顶部和底部的渐变宽度。
    noise_ratio : float
        替换用高斯噪声的强度 = 数据 RMS × noise_ratio。
    nfft_t_factor : int
        时间轴 FFT 点数 = nfft_t_factor × nt（补零倍数），建议 2。
    nfft_x_factor : int
        空间轴 FFT 点数 = nfft_x_factor × nx，建议 2。
    """

    def __init__(
        self,
        dt=0.004,
        dx=25,
        v_min=100,
        v_max=1200,
        f_max_noise=25.0,
        f_min_noise=0.0,
        taper_v=50.0,
        taper_f=3.0,
        noise_ratio=0.1,
        nfft_t_factor=2,
        nfft_x_factor=2,
    ):
        self.dt = dt
        self.dx = dx
        self.v_min = v_min
        self.v_max = v_max
        self.f_max_noise = f_max_noise
        self.f_min_noise = f_min_noise
        self.taper_v = taper_v
        self.taper_f = taper_f
        self.noise_ratio = noise_ratio
        self.nfft_t_factor = nfft_t_factor
        self.nfft_x_factor = nfft_x_factor

        print("FKMaskGenerator 初始化")
        print(f"  面波速度范围 : {v_min}–{v_max} m/s")
        print(f"  面波频率范围 : {f_min_noise}–{f_max_noise} Hz")
        print(f"  速度过渡带   : ±{taper_v} m/s")
        print(f"  频率过渡带   : ±{taper_f} Hz")

    # ------------------------------------------------------------------
    # F-K 域楔形掩码构建
    # ------------------------------------------------------------------

    def _build_fk_mask(self, nt, nx):
        """
        在 F-K 域构建面波楔形软掩码。

        掩码为 1 的区域：面波占主导（将被高斯噪声替换）
        掩码为 0 的区域：有效信号（保留原始数据）

        楔形定义：
            |f| / |k| 在 [v_min, v_max] 之间（视速度范围）
            且 f 在 [f_min_noise, f_max_noise] 之间（面波频带）

        边界均做 sigmoid 软过渡，消除截断伪影。
        """
        nfft_t = int(2 ** np.ceil(np.log2(nt * self.nfft_t_factor)))
        nfft_x = int(2 ** np.ceil(np.log2(nx * self.nfft_x_factor)))

        # 频率轴和波数轴（fftshift 后）
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft_t, d=self.dt))  # Hz
        kwave = np.fft.fftshift(np.fft.fftfreq(nfft_x, d=self.dx))  # 1/m

        F, K = np.meshgrid(freqs, kwave, indexing='ij')  # (nfft_t, nfft_x)

        # 视速度场：v_app = f / k（带符号，正负代表传播方向）
        with np.errstate(divide='ignore', invalid='ignore'):
            v_app = np.where(np.abs(K) > 1e-10, F / K, np.inf)

        abs_v = np.abs(v_app)
        abs_f = np.abs(F)

        # ---- sigmoid 软过渡函数 ----
        # 在边界附近用 sigmoid 平滑，宽度由 taper 控制
        def soft_step_up(x, center, width):
            """x > center 时趋向 1，宽度 width。"""
            return 1.0 / (1.0 + np.exp(-6.0 * (x - center) / (width + 1e-10)))

        def soft_step_down(x, center, width):
            """x < center 时趋向 1，宽度 width。"""
            return 1.0 / (1.0 + np.exp(6.0 * (x - center) / (width + 1e-10)))

        # ---- 速度边界 ----
        # 视速度 >= v_min（下边界）
        mask_v_low  = soft_step_down(abs_v, self.v_min, self.taper_v)
        # 视速度 <= v_max（上边界）
        mask_v_high = soft_step_up(abs_v, self.v_max, self.taper_v)
        # 两者取最小值（面波区域内两个条件都满足）
        mask_v = mask_v_low * mask_v_high

        # ---- 频率边界 ----
        mask_f_low  = soft_step_up(abs_f, self.f_min_noise, self.taper_f)
        mask_f_high = soft_step_down(abs_f, self.f_max_noise, self.taper_f)
        mask_f = mask_f_low * mask_f_high

        # ---- 最终 F-K 域掩码 ----
        fk_mask = mask_v * mask_f
        fk_mask = np.clip(fk_mask, 0.0, 1.0)

        return fk_mask, freqs, kwave, nfft_t, nfft_x

    # ------------------------------------------------------------------
    # F-K 掩码 → 时空域软掩码
    # ------------------------------------------------------------------

    def _fk_to_tx_mask(self, fk_mask, nt, nx, nfft_t, nfft_x):
        """
        将 F-K 域掩码通过逆 FFT 转换到时空域。

        F-K 域掩码本身在频率域是连续的，
        逆变换后在时空域也是空间连续的，不存在碎斑。

        逆变换后取绝对值并归一化到 [0, 1]。
        """
        # ifftshift 还原到 FFT 约定的频率顺序
        fk_ifftshift = np.fft.ifftshift(fk_mask)

        # 逆 FFT
        tx_mask_full = np.real(np.fft.ifft2(fk_ifftshift))

        # 裁剪回原始数据尺寸（去掉补零部分）
        tx_mask = tx_mask_full[:nt, :nx]

        # 归一化到 [0, 1]
        tx_mask = np.abs(tx_mask)
        tx_min  = tx_mask.min()
        tx_max  = tx_mask.max()
        if tx_max - tx_min > 1e-10:
            tx_mask = (tx_mask - tx_min) / (tx_max - tx_min)

        return tx_mask

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def compute_fk_spectrum(self, data):
        """计算 F-K 振幅谱，用于可视化和参数调试。"""
        nt, nx = data.shape
        nfft_t = int(2 ** np.ceil(np.log2(nt * self.nfft_t_factor)))
        nfft_x = int(2 ** np.ceil(np.log2(nx * self.nfft_x_factor)))
        FK = np.fft.fftshift(np.abs(np.fft.fft2(data, s=(nfft_t, nfft_x))))
        FK_db = 20 * np.log10(FK / (FK.max() + 1e-10) + 1e-6)
        freqs  = np.fft.fftshift(np.fft.fftfreq(nfft_t, d=self.dt))
        kwave  = np.fft.fftshift(np.fft.fftfreq(nfft_x, d=self.dx))
        return FK_db, freqs, kwave

    def generate_mask(self, data):
        """
        生成时空域软掩码 α(t, x)。

        Parameters
        ----------
        data : ndarray, shape (nt, nx)

        Returns
        -------
        alpha : ndarray, shape (nt, nx)
            掩码强度，0=保留原始，1=完全替换。
        fk_mask : ndarray
            F-K 域掩码（供可视化）。
        meta : dict
            freqs, kwave, nfft_t, nfft_x。
        """
        nt, nx = data.shape
        print(f"  构建 F-K 域楔形掩码...")
        fk_mask, freqs, kwave, nfft_t, nfft_x = self._build_fk_mask(nt, nx)

        print(f"  F-K → 时空域逆变换...")
        alpha = self._fk_to_tx_mask(fk_mask, nt, nx, nfft_t, nfft_x)

        meta = dict(freqs=freqs, kwave=kwave,
                    nfft_t=nfft_t, nfft_x=nfft_x)
        return alpha, fk_mask, meta

    def apply_mask(self, data, seed=None):
        """
        生成掩码并应用到数据。

        Parameters
        ----------
        data : ndarray, shape (nt, nx)
        seed : int or None

        Returns
        -------
        masked_data : ndarray   DMSSL 网络输入
        alpha : ndarray         掩码强度
        fk_mask : ndarray       F-K 域掩码（供可视化）
        meta : dict
        """
        if seed is not None:
            np.random.seed(seed)

        alpha, fk_mask, meta = self.generate_mask(data)

        std   = np.sqrt(np.mean(data ** 2)) * self.noise_ratio
        noise = np.random.randn(*data.shape) * std

        masked_data = data * (1.0 - alpha) + noise * alpha
        return masked_data, alpha, fk_mask, meta


# ===========================================================================
# 可视化
# ===========================================================================

def plot_fk_mask_overview(data, fk_db, freqs, kwave,
                           fk_mask, alpha, masked_data,
                           dt=0.004, dx=25, save_path=None):
    """六图总览：F-K谱 | F-K掩码 | 叠加 | 时空掩码α | 原始 | 掩码后"""
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
    t_max, x_max = nt*dt, nx*dx
    tx_ext = [0, x_max, t_max, 0]

    # F-K 轴范围（只显示正频率部分）
    f_pos  = freqs[freqs >= 0]
    k_half = np.abs(kwave).max()
    fk_ext = [-k_half, k_half, f_pos.max(), 0]

    fk_pos     = fk_db[freqs >= 0, :]
    fkmask_pos = fk_mask  # fk_mask 已经是正频率部分

    def ax_imshow(pos, d, title, cmap, vmin, vmax, ext, xlabel, ylabel, clabel=""):
        ax = fig.add_subplot(gs[pos])
        im = ax.imshow(d, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=ext)
        ax.set_title(title, fontsize=10, color="#58A6FF", pad=5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(clabel, fontsize=7, color="#E6EDF3")
        cb.ax.tick_params(colors="#E6EDF3", labelsize=7)
        return ax

    ax_imshow((0,0), fk_pos,      "F-K 振幅谱 (dB)",
              "inferno", -60, 0, fk_ext,
              "波数 (1/m)", "频率 (Hz)", "dB")

    ax_imshow((0,1), fkmask_pos,  "F-K 域楔形掩码",
              "Blues", 0, 1, fk_ext,
              "波数 (1/m)", "频率 (Hz)", "强度")

    # 叠加显示
    ax3 = fig.add_subplot(gs[0,2])
    ax3.imshow(fk_pos, aspect="auto", cmap="inferno",
               vmin=-60, vmax=0, extent=fk_ext, alpha=0.8)
    ax3.imshow(fkmask_pos, aspect="auto", cmap="Blues",
               vmin=0, vmax=1, extent=fk_ext, alpha=0.45)
    ax3.set_title("F-K 谱 + 掩码叠加", fontsize=10, color="#58A6FF", pad=5)
    ax3.set_xlabel("波数 (1/m)", fontsize=8)
    ax3.set_ylabel("频率 (Hz)",  fontsize=8)

    ax_imshow((1,0), alpha,       "时空域掩码强度 alpha",
              "Blues", 0, 1, tx_ext,
              "偏移距 (m)", "时间 (s)", "alpha")

    vmax = np.percentile(np.abs(data), 95)
    ax_imshow((1,1), data,        "原始炮集",
              "seismic", -vmax, vmax, tx_ext,
              "偏移距 (m)", "时间 (s)", "振幅")

    ax_imshow((1,2), masked_data, "DMSSL 输入（掩码后）",
              "seismic", -vmax, vmax, tx_ext,
              "偏移距 (m)", "时间 (s)", "振幅")

    fig.suptitle("方案一：F-K 域自动掩码", fontsize=13,
                 color="#58A6FF", y=0.99)
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  已保存：{save_path}")
    return fig


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == "__main__":

    input_path = Path(r"D:\桌面\面波压制\深层数据面波压制\最终数据\original.npy")
    output_dir = Path.cwd() / "mask_fk_output"
    output_dir.mkdir(exist_ok=True)

    dt, dx = 0.004, 25

    # 参数设置
    # v_min/v_max 从速度拾取工具读取的面波速度范围
    # f_max_noise 从频谱图读取的面波截止频率
    # taper_v/taper_f 控制软边界宽度，建议先用默认值
    fk_params = dict(
        v_min        = 100,
        v_max        = 1200,
        f_max_noise  = 25.0,
        f_min_noise  = 0.0,
        taper_v      = 80.0,
        taper_f      = 3.0,
        noise_ratio  = 0.1,
    )

    print("=" * 55)
    print("方案一：F-K 域自动掩码生成")
    print("=" * 55)

    data = np.load(str(input_path)).astype(float).squeeze()
    assert data.ndim == 2
    nt, nx = data.shape
    print(f"数据尺寸：{nt} x {nx}")

    gen = FKMaskGenerator(**fk_params, dt=dt, dx=dx)

    print("\n生成掩码...")
    masked_data, alpha, fk_mask, meta = gen.apply_mask(data, seed=42)
    freqs = meta['freqs']
    kwave = meta['kwave']

    # 计算 F-K 谱（使用与掩码一致的 FFT 参数）
    nfft_t = meta['nfft_t']
    nfft_x = meta['nfft_x']
    FK = np.fft.fftshift(np.abs(np.fft.fft2(data, s=(nfft_t, nfft_x))))
    fk_db = 20 * np.log10(FK / (FK.max() + 1e-10) + 1e-6)

    np.save(str(output_dir / "fk_alpha.npy"),       alpha)
    np.save(str(output_dir / "fk_masked_data.npy"), masked_data)
    print(f"掩码覆盖率：{alpha.mean()*100:.1f}%")
    print(f"掩码连续性（过渡区比例）："
          f"{((alpha>0.05)&(alpha<0.95)).mean()*100:.1f}%")

    print("\n生成可视化...")
    plot_fk_mask_overview(
        data, fk_db, freqs, kwave,
        fk_mask[freqs >= 0, :],   # 正频率部分
        alpha, masked_data,
        dt=dt, dx=dx,
        save_path=output_dir / "fk_mask_overview.png"
    )
    print(f"\n完成，结果保存至：{output_dir}")