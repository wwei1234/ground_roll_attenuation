"""
梯度流正则化（GFR）面波压制方法
==================================

复现论文：
    Cai & Ma (2018), "Ground rolls attenuation via gradient flow regularization",
    Journal of Applied Geophysics, 155, 246-255.

依赖：numpy, scipy, matplotlib
运行：python gfr.py
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# ===========================================================================
# 工具函数：频率滤波
# ===========================================================================

def bandpass_filter(data, dt, f_low=None, f_high=None):
    """
    对数据施加低通或高通滤波（频域直接裁切，加汉宁窗平滑过渡）。

    Parameters
    ----------
    data  : ndarray, shape (nt, nx)
    dt    : float，时间采样间隔 (s)
    f_low : float or None，低通截止频率 (Hz)，None 表示不施加低通
    f_high: float or None，高通截止频率 (Hz)，None 表示不施加高通

    Returns
    -------
    filtered : ndarray, shape (nt, nx)
    """
    nt = data.shape[0]
    freqs = np.fft.rfftfreq(nt, d=dt)   # 单边频率轴

    # 构建频率域窗函数
    win = np.ones(len(freqs))

    if f_low is not None:
        # 低通：f > f_low 部分逐渐衰减
        idx = np.searchsorted(freqs, f_low)
        taper = np.zeros(len(freqs))
        taper[:idx] = 1.0
        # 过渡带（半个汉宁窗）：idx 前后各 max(2,idx//4) 个点
        tw = max(2, idx // 4)
        for k in range(tw):
            taper[max(0, idx - tw + k)] = 0.5 * (
                1 - np.cos(np.pi * k / tw)
            )
        win *= taper

    if f_high is not None:
        # 高通：f < f_high 部分逐渐衰减
        idx = np.searchsorted(freqs, f_high)
        taper = np.zeros(len(freqs))
        taper[idx:] = 1.0
        tw = max(2, idx // 4)
        for k in range(tw):
            taper[min(len(freqs)-1, idx + k)] = 0.5 * (
                1 - np.cos(np.pi * (tw - k) / tw)
            )
        win *= taper

    # 沿时间轴做 rfft，乘窗，irfft
    DATA = np.fft.rfft(data, axis=0)
    DATA *= win[:, np.newaxis]
    return np.fft.irfft(DATA, n=nt, axis=0)


# ===========================================================================
# 核心：结构张量法计算梯度流
# ===========================================================================

def compute_gradient_flow(data, sigma_grad=1.0, sigma_tensor=3.0):
    """
    用结构张量方法计算数据每点的梯度主方向（单位向量），
    即梯度流张量 G（shape: nt × nx × 2）。

    流程
    ----
    1. 对 data 做高斯平滑（减少噪声对梯度的影响）
    2. 计算 x 方向和 t 方向的梯度（前向差分）
    3. 在局部窗口内统计梯度的协方差矩阵（结构张量）
    4. 取结构张量最大特征值对应的特征向量作为梯度主方向
    5. 归一化为单位向量

    Parameters
    ----------
    data        : ndarray, shape (nt, nx)
    sigma_grad  : float，计算梯度前的高斯平滑标准差（采样点）
    sigma_tensor: float，结构张量局部平均的高斯核标准差

    Returns
    -------
    G : ndarray, shape (nt, nx, 2)
        G[i, j, 0]：x（空间）方向分量
        G[i, j, 1]：t（时间）方向分量
    """
    nt, nx = data.shape

    # 1. 高斯平滑
    smoothed = gaussian_filter(data, sigma=sigma_grad)

    # 2. 梯度（中心差分，边界用前向/后向差分）
    gx = np.gradient(smoothed, axis=1)   # 空间方向
    gt = np.gradient(smoothed, axis=0)   # 时间方向

    # 3. 结构张量各分量：局部窗口内的加权平均
    Jxx = gaussian_filter(gx * gx, sigma=sigma_tensor)
    Jxt = gaussian_filter(gx * gt, sigma=sigma_tensor)
    Jtt = gaussian_filter(gt * gt, sigma=sigma_tensor)

    # 4. 逐点求 2×2 结构张量的主方向
    G = np.zeros((nt, nx, 2))

    # 利用 2×2 对称矩阵特征值公式的向量化版本
    # J = [[Jxx, Jxt], [Jxt, Jtt]]
    # 特征值：λ = (trace ± sqrt(trace² - 4*det)) / 2
    trace = Jxx + Jtt
    det   = Jxx * Jtt - Jxt * Jxt
    disc  = np.sqrt(np.maximum((trace**2 / 4) - det, 0))

    lam_max = trace / 2 + disc   # 最大特征值

    # 对应最大特征值的特征向量：(J - lam_min * I) 的列
    # 等价于 (Jxx - lam_min, Jxt) 的方向
    lam_min = trace / 2 - disc
    vx = Jxx - lam_min       # 特征向量 x 分量（未归一化）
    vt = Jxt                  # 特征向量 t 分量

    # 当 vx 和 vt 都接近 0（均匀区域），退化到 (1, 0)
    norm = np.sqrt(vx**2 + vt**2)
    degenerate = norm < 1e-10
    vx[degenerate] = 1.0
    vt[degenerate] = 0.0
    norm[degenerate] = 1.0

    G[:, :, 0] = vx / norm
    G[:, :, 1] = vt / norm

    return G


# ===========================================================================
# 核心：DFT 加速的 BCD 求解器
# ===========================================================================

def _build_dtd_spectrum(nt, nx):
    """
    预计算 D^T D 在 2D DFT 域的对角化表示（仅需计算一次）。

    在周期边界条件下，前向差分矩阵 D_x 和 D_t 均为循环矩阵，
    D^T D = D_x^T D_x + D_t^T D_t 也是循环矩阵，
    可由 2D DFT 对角化：

        F (D^T D) F^H = diag(eigenvalues)

    特征值等于 D^T D 第一行在 DFT 域的变换。
    前向差分算子 d = [−1, 1, 0, ..., 0]，
    D^T D 的特征值为 2 − 2cos(2πk/n)。
    """
    # 时间轴特征值
    kt = np.fft.fftfreq(nt) * nt          # k = 0, 1, ..., nt-1
    ev_t = 2 - 2 * np.cos(2 * np.pi * kt / nt)

    # 空间轴特征值
    kx = np.fft.fftfreq(nx) * nx
    ev_x = 2 - 2 * np.cos(2 * np.pi * kx / nx)

    # D^T D 的 2D 特征值（广播）
    ev_dtd = ev_t[:, np.newaxis] + ev_x[np.newaxis, :]

    return ev_dtd   # shape (nt, nx)


def _apply_gradient_operator(X):
    """
    计算前向差分梯度 D(X)，返回 (gx, gt) 两个 nt×nx 矩阵。
    使用 np.roll 实现周期边界的前向差分。
    """
    gx = np.roll(X, -1, axis=1) - X   # 空间方向
    gt = np.roll(X, -1, axis=0) - X   # 时间方向
    return gx, gt


def _apply_dtd(X):
    """
    计算 D^T D X（周期边界条件下的拉普拉斯算子）。
    D^T = 后向差分算子，D^T D X = X - roll(X,1) - roll(X,-1) + ...
    等价于：D^T (D X)
    """
    gx, gt = _apply_gradient_operator(X)
    # D^T (gx) = gx - roll(gx, 1, axis=1)
    dtgx = gx - np.roll(gx, 1, axis=1)
    dtgt = gt - np.roll(gt, 1, axis=0)
    return dtgx + dtgt


def _dt_diag_g_s(G, S):
    """
    计算 D^T (diag(G) S)。

    G : (nt, nx, 2)  梯度流（单位向量）
    S : (nt, nx)     缩放场（两方向分量相等）

    diag(G) S 的 x 分量 = G[:,:,0] * S
    diag(G) S 的 t 分量 = G[:,:,1] * S

    D^T [(sx, st)] = D_x^T sx + D_t^T st
    """
    sx = G[:, :, 0] * S   # x 方向
    st = G[:, :, 1] * S   # t 方向

    # D_x^T sx = sx - roll(sx, 1, axis=1)
    dtsx = sx - np.roll(sx, 1, axis=1)
    # D_t^T st = st - roll(st, 1, axis=0)
    dtst = st - np.roll(st, 1, axis=0)

    return dtsx + dtst


def update_X(X_other, G, S, lam, ev_dtd, X_raw):
    """
    更新 X_s 或 X_g（对称，传入对方变量）。

    求解：(λ⁻¹ I + D^T D) X = λ⁻¹ (X_raw - X_other) + D^T diag(G) S

    用 DFT 对角化求解：
        X = F⁻¹ [ F(RHS) / F(λ⁻¹ I + D^T D) ]
    """
    RHS = (X_raw - X_other) / lam + _dt_diag_g_s(G, S)

    # 分母：λ⁻¹ + ev_dtd（逐点除法）
    denom = 1.0 / lam + ev_dtd

    X_new = np.real(np.fft.ifft2(np.fft.fft2(RHS) / denom))
    return X_new


def update_S(X, G):
    """
    更新 S_s 或 S_g。

    解析解：将 D(X) 投影到梯度流 G 上。
    S_{i,j} = <G_{i,j,:}, D(X)_{i,j,:}> / <G_{i,j,:}, G_{i,j,:}>

    由于 G 是单位向量，分母恒为 1，化简为内积。
    """
    gx, gt = _apply_gradient_operator(X)

    # 内积：G[:,:,0]*gx + G[:,:,1]*gt
    inner = G[:, :, 0] * gx + G[:, :, 1] * gt

    # 分母：||G||^2 = G0^2 + G1^2 = 1（单位向量），但保险起见显式计算
    denom = G[:, :, 0]**2 + G[:, :, 1]**2
    denom = np.maximum(denom, 1e-10)

    S_new = inner / denom
    return S_new


# ===========================================================================
# GFR 主类
# ===========================================================================

class GFR:
    """
    梯度流正则化（GFR）面波压制方法。

    Parameters
    ----------
    dt : float
        时间采样间隔 (s)。
    f_low : float
        低通滤波截止频率 (Hz)，用于提取面波低频参考。
        建议 3–7 Hz，只需大致避开频率重叠区。
    f_high : float
        高通滤波截止频率 (Hz)，用于提取反射波高频参考。
        建议 20–40 Hz，只需大致避开频率重叠区。
    lambda1 : float
        反射波梯度流正则化权重。
        越大：反射波越严格遵循参考梯度方向，
        同相轴更连续但可能损失部分信号能量。
    lambda2 : float
        面波梯度流正则化权重。
        越大：面波越严格遵循参考梯度方向，
        面波压制更彻底但可能引入轻微伪影。
    n_iter : int
        BCD 迭代次数，通常 50–200 次已收敛。
    sigma_grad : float
        计算梯度流时的预平滑高斯核标准差（采样点）。
    sigma_tensor : float
        结构张量局部平均的高斯核标准差（采样点）。
    """

    def __init__(
        self,
        dt=0.002,
        f_low=5.0,
        f_high=30.0,
        lambda1=0.5,
        lambda2=0.5,
        n_iter=100,
        sigma_grad=1.0,
        sigma_tensor=3.0,
    ):
        self.dt           = dt
        self.f_low        = f_low
        self.f_high       = f_high
        self.lambda1      = lambda1
        self.lambda2      = lambda2
        self.n_iter       = n_iter
        self.sigma_grad   = sigma_grad
        self.sigma_tensor = sigma_tensor

        print("GFR 参数初始化")
        print(f"  低通截止频率  : {f_low} Hz（面波参考）")
        print(f"  高通截止频率  : {f_high} Hz（反射波参考）")
        print(f"  正则化参数    : λ1={lambda1}, λ2={lambda2}")
        print(f"  迭代次数      : {n_iter}")

    def _check_convergence(self, Xs_new, Xs_old, Xg_new, Xg_old):
        """计算相对变化量，用于监控收敛。"""
        dXs = np.linalg.norm(Xs_new - Xs_old) / (np.linalg.norm(Xs_old) + 1e-10)
        dXg = np.linalg.norm(Xg_new - Xg_old) / (np.linalg.norm(Xg_old) + 1e-10)
        return max(dXs, dXg)

    def separate(self, data, verbose=True):
        """
        执行 GFR 面波压制。

        Parameters
        ----------
        data    : ndarray, shape (nt, nx)
        verbose : bool，是否打印迭代进度

        Returns
        -------
        Xs : ndarray, shape (nt, nx)  分离的反射波
        Xg : ndarray, shape (nt, nx)  分离的面波（被去除的成分）
        info : dict
            low_freq   : 低频参考数据
            high_freq  : 高频参考数据
            Gg         : 面波梯度流
            Gs         : 反射波梯度流
            residuals  : 各迭代的相对变化量
        """
        nt, nx = data.shape
        data   = data.astype(np.float64)

        # -----------------------------------------------------------
        # 第一步：频率分离，获取梯度流参考
        # -----------------------------------------------------------
        if verbose:
            print(f"\n开始 GFR 处理：{nt} × {nx}")
            print("步骤 1：频率分离...")

        Xg0 = bandpass_filter(data, self.dt, f_low=self.f_low)
        Xs0 = bandpass_filter(data, self.dt, f_high=self.f_high)

        # -----------------------------------------------------------
        # 第二步：计算梯度流
        # -----------------------------------------------------------
        if verbose:
            print("步骤 2：计算梯度流（结构张量法）...")

        Gg = compute_gradient_flow(Xg0,
                                   sigma_grad=self.sigma_grad,
                                   sigma_tensor=self.sigma_tensor)
        Gs = compute_gradient_flow(Xs0,
                                   sigma_grad=self.sigma_grad,
                                   sigma_tensor=self.sigma_tensor)

        # -----------------------------------------------------------
        # 第三步：预计算 DFT 对角化分母（只算一次）
        # -----------------------------------------------------------
        if verbose:
            print("步骤 3：预计算 DFT 对角化分母...")

        ev_dtd = _build_dtd_spectrum(nt, nx)

        # -----------------------------------------------------------
        # 第四步：BCD 迭代
        # -----------------------------------------------------------
        if verbose:
            print(f"步骤 4：BCD 迭代（共 {self.n_iter} 次）...")

        # 初始化
        Xs = np.zeros((nt, nx))
        Xg = np.zeros((nt, nx))
        Ss = np.zeros((nt, nx))
        Sg = np.zeros((nt, nx))

        residuals = []
        log_interval = max(1, self.n_iter // 10)

        for it in range(self.n_iter):
            Xs_old = Xs.copy()
            Xg_old = Xg.copy()

            # 1. 更新 Xs
            Xs = update_X(Xg, Gs, Ss, self.lambda1, ev_dtd, data)

            # 2. 更新 Ss
            Ss = update_S(Xs, Gs)

            # 3. 更新 Xg
            Xg = update_X(Xs, Gg, Sg, self.lambda2, ev_dtd, data)

            # 4. 更新 Sg
            Sg = update_S(Xg, Gg)

            # 监控收敛
            res = self._check_convergence(Xs, Xs_old, Xg, Xg_old)
            residuals.append(res)

            if verbose and (it + 1) % log_interval == 0:
                print(f"  迭代 {it+1:4d}/{self.n_iter}  "
                      f"相对变化量 = {res:.2e}")

        if verbose:
            print(f"迭代完成，最终相对变化量 = {residuals[-1]:.2e}")

        info = dict(
            low_freq  = Xg0,
            high_freq = Xs0,
            Gg        = Gg,
            Gs        = Gs,
            residuals = residuals,
        )

        return Xs, Xg, info


# ===========================================================================
# 可视化
# ===========================================================================

def plot_gfr_results(data, Xs, Xg, info, dt=0.002, dx=10, save_path=None):
    """
    六图总览：
    原始数据 | 高频参考 | 低频参考 | 分离反射波 | 去除面波 | 收敛曲线
    """
    fig = plt.figure(figsize=(22, 9), facecolor="#0D1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.05, right=0.97,
                            top=0.93, bottom=0.08,
                            wspace=0.25, hspace=0.35)
    plt.rcParams.update({
        "text.color":"#E6EDF3","axes.labelcolor":"#E6EDF3",
        "xtick.color":"#E6EDF3","ytick.color":"#E6EDF3",
        "axes.edgecolor":"#30363D","axes.facecolor":"#161B22",
    })

    nt, nx = data.shape
    ext  = [0, nx*dx, nt*dt, 0]
    vmax = np.percentile(np.abs(data), 95)

    def axi(pos, d, title, vscale=1.0):
        ax = fig.add_subplot(gs[pos])
        vm = vmax * vscale
        im = ax.imshow(d, aspect="auto", cmap="seismic",
                       vmin=-vm, vmax=vm, extent=ext)
        ax.set_title(title, fontsize=10, color="#58A6FF", pad=5)
        ax.set_xlabel("偏移距 (m)", fontsize=8)
        ax.set_ylabel("时间 (s)",   fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    axi((0,0), data,         "原始数据")
    axi((0,1), info["high_freq"], f"高频参考 (>{info.get('f_high',30)} Hz)", 0.5)
    axi((0,2), info["low_freq"],  f"低频参考 (<{info.get('f_low',5)} Hz)",   0.5)
    axi((1,0), Xs,           "GFR 分离：反射波")
    axi((1,1), Xg,           "GFR 去除：面波",   0.5)

    # 收敛曲线
    ax_conv = fig.add_subplot(gs[1,2])
    ax_conv.semilogy(info["residuals"], color="#58A6FF", lw=1.5)
    ax_conv.set_title("收敛曲线（相对变化量）", fontsize=10,
                      color="#58A6FF", pad=5)
    ax_conv.set_xlabel("迭代次数", fontsize=8)
    ax_conv.set_ylabel("相对变化量", fontsize=8)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.set_facecolor("#161B22")

    fig.suptitle("GFR 梯度流正则化面波压制结果",
                 fontsize=13, color="#58A6FF", y=0.99)

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  图片已保存：{save_path}")
    return fig


def plot_spectrum_comparison(data, Xs, Xg, dt=0.002, save_path=None):
    """频谱对比：原始 / 分离反射波 / 去除面波。"""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0D1117")
    plt.rcParams.update({
        "text.color":"#E6EDF3","axes.labelcolor":"#E6EDF3",
        "xtick.color":"#E6EDF3","ytick.color":"#E6EDF3",
        "axes.edgecolor":"#30363D","axes.facecolor":"#161B22",
    })
    ax.set_facecolor("#161B22")

    freqs = np.fft.rfftfreq(data.shape[0], d=dt)

    def mean_spectrum(d):
        return np.mean(np.abs(np.fft.rfft(d, axis=0)), axis=1)

    ax.plot(freqs, mean_spectrum(data), color="black",  lw=1.5, label="原始数据")
    ax.plot(freqs, mean_spectrum(Xs),   color="#58A6FF",lw=1.5, label="GFR 反射波")
    ax.plot(freqs, mean_spectrum(Xg),   color="#FF6B6B",lw=1.5, label="GFR 面波")
    ax.set_xlabel("频率 (Hz)", fontsize=10)
    ax.set_ylabel("平均振幅",  fontsize=10)
    ax.set_title("平均频谱对比", fontsize=12, color="#58A6FF")
    ax.legend(fontsize=9, facecolor="#161B22", edgecolor="#30363D")
    ax.set_xlim(0, min(freqs.max(), 100))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  图片已保存：{save_path}")
    return fig


def plot_gradient_flow(Gg, Gs, dt=0.002, dx=10,
                       subsample=8, save_path=None):
    """
    可视化梯度流方向场（箭头图）。
    subsample : 每隔多少点画一个箭头（降采样，避免过密）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0D1117")
    plt.rcParams.update({
        "text.color":"#E6EDF3","axes.labelcolor":"#E6EDF3",
        "xtick.color":"#E6EDF3","ytick.color":"#E6EDF3",
        "axes.edgecolor":"#30363D","axes.facecolor":"#161B22",
    })

    nt, nx = Gg.shape[:2]
    t_idx = np.arange(0, nt, subsample)
    x_idx = np.arange(0, nx, subsample)
    T, X  = np.meshgrid(t_idx * dt, x_idx * dx, indexing="ij")

    for ax, G, title in zip(axes,
                              [Gs, Gg],
                              ["反射波梯度流 Gs", "面波梯度流 Gg"]):
        ax.set_facecolor("#161B22")
        Gx = G[np.ix_(t_idx, x_idx, [0])].squeeze()
        Gt = G[np.ix_(t_idx, x_idx, [1])].squeeze()
        ax.quiver(X, T, Gx, -Gt, color="#58A6FF", scale=30, alpha=0.7,
                  width=0.003)
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(nt * dt, 0)
        ax.set_title(title, fontsize=11, color="#58A6FF")
        ax.set_xlabel("偏移距 (m)", fontsize=9)
        ax.set_ylabel("时间 (s)",   fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  图片已保存：{save_path}")
    return fig


# ===========================================================================
# 合成数据生成（用于验证）
# ===========================================================================

def make_synthetic_data(nt=500, nx=60, dt=0.002, dx=10,
                         noise_level=0.05, seed=42):
    """
    生成含线性面波的合成炮集。

    反射波：双曲线同相轴（Ricker 子波）
    面波：线性同相轴，低频（5–20 Hz），低速，大振幅
    随机噪声：带限高斯噪声
    """
    np.random.seed(seed)
    t = np.arange(nt) * dt
    data = np.zeros((nt, nx))

    # --- 反射波（3 条双曲线）---
    v_nmo = 2000.0   # NMO 速度
    offsets = np.arange(nx) * dx
    for t0 in [0.3, 0.6, 0.9]:
        for j, x in enumerate(offsets):
            t_hyp = np.sqrt(t0**2 + (x / v_nmo)**2)
            # Ricker 子波
            f0 = 40.0
            tau = t - t_hyp
            ricker = (1 - 2 * (np.pi * f0 * tau)**2) * \
                      np.exp(-(np.pi * f0 * tau)**2)
            data[:, j] += 0.3 * ricker

    # --- 面波（3 组线性同相轴）---
    for v_gr, amp, f_gr in [(300, 1.0, 8), (500, 0.8, 12), (200, 0.6, 6)]:
        for j, x in enumerate(offsets):
            t_gr = x / v_gr
            tau  = t - t_gr
            # 低频正弦调制高斯包络
            grw = amp * np.exp(-(tau)**2 / (2 * 0.04**2)) * \
                  np.sin(2 * np.pi * f_gr * t)
            data[:, j] += grw

    # --- 随机噪声 ---
    raw_noise = np.random.randn(nt, nx)
    noise = bandpass_filter(raw_noise, dt, f_low=80)
    data += noise_level * noise / (np.std(noise) + 1e-10) * np.std(data)

    return data


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == "__main__":

    output_dir = Path.cwd() / "gfr_output"
    output_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------
    # 参数设置
    # -------------------------------------------------------
    USE_REAL_DATA = True   # True = 读取 .npy 文件；False = 合成数据

    if USE_REAL_DATA:
        input_path = Path(
            r"D:\桌面\面波压制\深层数据面波压制\最终数据\original.npy"
        )
        dt, dx = 0.004, 25
        data = np.load(str(input_path)).astype(float).squeeze()
        assert data.ndim == 2
    else:
        dt, dx = 0.002, 10
        print("使用合成数据进行验证...")
        data = make_synthetic_data(nt=500, nx=60, dt=dt, dx=dx)

    nt, nx = data.shape
    print(f"数据尺寸：{nt} × {nx}")

    # GFR 参数
    # f_low/f_high：只需大致避开频率重叠区
    # lambda1/lambda2：0.1–1.0，较大值增强正则化
    # n_iter：100 通常足够
    gfr_params = dict(
        dt           = dt,
        f_low        = 8.0,
        f_high       = 15.0,
        lambda1      = 0.3,
        lambda2      = 0.5,
        n_iter       = 100,
        sigma_grad   = 1.0,
        sigma_tensor = 3.0,
    )
    # -------------------------------------------------------

    print("\n" + "="*55)
    print("GFR 梯度流正则化面波压制")
    print("="*55)

    gfr = GFR(**gfr_params)
    Xs, Xg, info = gfr.separate(data, verbose=True)

    # 把频率参数写入 info 供可视化使用
    info["f_low"]  = gfr.f_low
    info["f_high"] = gfr.f_high

    # 保存结果
    np.save(str(output_dir / "gfr_signal.npy"), Xs)
    np.save(str(output_dir / "gfr_groundroll.npy"), Xg)
    print(f"\n结果已保存至：{output_dir}")

    # 可视化
    print("\n生成可视化图片...")
    plot_gfr_results(
        data, Xs, Xg, info, dt=dt, dx=dx,
        save_path=output_dir / "gfr_overview.png"
    )
    plot_spectrum_comparison(
        data, Xs, Xg, dt=dt,
        save_path=output_dir / "gfr_spectrum.png"
    )
    plot_gradient_flow(
        info["Gg"], info["Gs"], dt=dt, dx=dx,
        subsample=max(1, nt // 30),
        save_path=output_dir / "gfr_gradient_flow.png"
    )

    # 简单定量评估
    print("\n定量评估：")
    snr_in  = 10 * np.log10(np.var(data) /
                             (np.var(data - Xs) + 1e-10))
    print(f"  输入数据方差      : {np.var(data):.4f}")
    print(f"  分离反射波方差    : {np.var(Xs):.4f}")
    print(f"  去除面波方差      : {np.var(Xg):.4f}")
    print(f"  数据保真度（残差）: {np.var(data - Xs - Xg):.6f}")
    print(f"  收敛迭代数        : {len(info['residuals'])}")
    print(f"  最终相对变化量    : {info['residuals'][-1]:.2e}")

    print("\n" + "="*55)
    print("GFR 处理完成")
    print("="*55)