from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

class LocalNonlinearFilter:
    """
    局部非线性滤波（LNF）方法，用于地震数据中的相干噪声（面波）压制。

    方法参考：
    Yuan et al. (2022), "Adaptive ground-roll attenuation using local nonlinear filtering",
    Computers & Geosciences, 164, 105124.
    """

    def __init__(
        self,
        f1=0,                  # 低通滤波通带起始频率 (Hz)
        f2=25,                 # 低通滤波截止频率 (Hz)
        dt=0.002,              # 时间采样间隔 (s)
        dx=10,                 # 道间距 (m)
        v_min=100,             # 最小扫描视速度 (m/s)
        v_max=2000,            # 最大扫描视速度 (m/s)
        dv=50,                 # 速度扫描间隔 (m/s)
        win_t=0.1,             # 时间窗口长度 (s)
        win_x=5,               # 空间窗口道数
        c1=0.6,                # 加权修正下阈值
        c2=0.8,                # 加权修正上阈值
        p=2,                   # 混合中值-均值滤波邻域参数
    ):
        """
        初始化 LNF 参数。

        Parameters
        ----------
        f1, f2 : float
            低通滤波器的频率范围，单位 Hz。
        dt : float
            时间采样间隔，单位 s。
        dx : float
            道间距，单位 m。
        v_min, v_max : float
            相干噪声视速度扫描范围，单位 m/s。
        dv : float
            速度扫描步长，单位 m/s。
        win_t : float
            处理时间窗口长度，单位 s。
        win_x : int
            空间窗口道数。若输入为偶数，会自动加 1 转成奇数。
        c1, c2 : float
            加权修正阈值，要求 0 < c1 < c2 < 1。
        p : int
            混合中值-均值滤波中，中值两侧各取 p 个相邻点。
        """
        self.f1 = f1
        self.f2 = f2
        self.dt = dt
        self.dx = dx
        self.v_min = v_min
        self.v_max = v_max
        self.dv = dv
        self.win_t = win_t
        self.win_x = win_x if win_x % 2 == 1 else win_x + 1
        self.c1 = c1
        self.c2 = c2
        self.p = p

        # 由输入参数计算处理窗口和速度扫描参数。
        self.N = int(win_t / dt)
        self.M = self.win_x
        self.Nv = int((v_max - v_min) / dv)
        if self.Nv <= 0:
            self.Nv = 1
        self.v_list = np.linspace(v_min, v_max, self.Nv)

        # 时间窗口点数保持为奇数，便于取中心点。
        if self.N % 2 == 0:
            self.N += 1

        print("LNF 参数初始化完成")
        print(f"  频率范围: {f1}-{f2} Hz")
        print(f"  速度扫描: {v_min}-{v_max} m/s，步长 {dv} m/s，共 {self.Nv} 个速度")
        print(f"  处理窗口: {self.N} 个采样点 x {self.M} 道")
        print(f"  加权阈值: c1={c1}, c2={c2}")

    def triangular_convolution_filter(self, nt, dt, f1, f2):
        """
        构建三角波卷积边缘滤波器。

        该滤波器用于频率分离，提取低频面波主导的频带，同时减弱 Gibbs 效应。
        """
        t = np.arange(nt) * dt - (nt // 2) * dt
        t = t + 1e-10

        # 三角波卷积边缘滤波器：
        # L(f1,f2) = [4*sin(pi*(f2+f1)*t)/(pi*t)]
        #            * [sin^2(pi*(f2-f1)*t/2) / ((f2-f1)^2*pi^2*t^2)]
        term1 = 4 * np.sin(np.pi * (f2 + f1) * t) / (np.pi * t)

        df = f2 - f1
        if df < 1e-6:
            df = 1e-6

        term2_num = np.sin(np.pi * df * t / 2) ** 2
        term2_den = (df ** 2) * (np.pi ** 2) * (t ** 2)
        term2 = term2_num / term2_den

        h = term1 * term2
        h = h / np.sum(np.abs(h))
        return h

    def frequency_division(self, data):
        """
        频率分离处理。

        使用低通滤波提取相干面波主导的低频分量。
        """
        nt, nx = data.shape

        filter_len = min(101, nt // 2 * 2 + 1)
        h = self.triangular_convolution_filter(filter_len, self.dt, self.f1, self.f2)

        s_low = np.zeros_like(data)
        for ix in range(nx):
            s_low[:, ix] = np.convolve(data[:, ix], h, mode="same")

        return s_low

    def extract_window_data(self, s, i0, j0, vk):
        """
        沿给定视速度提取局部数据窗口。

        Parameters
        ----------
        s : ndarray, shape (nt, nx)
            低频分量数据。
        i0, j0 : int
            中心点位置，分别为时间采样点和道号。
        vk : float
            当前扫描视速度，单位 m/s。

        Returns
        -------
        w : ndarray, shape (N, M)
            沿视速度方向提取的局部数据窗口。
        valid : bool
            当前窗口是否完全位于数据范围内。
        """
        nt, nx = s.shape
        N, M = self.N, self.M

        w = np.zeros((N, M))
        half_N = N // 2
        half_M = M // 2

        valid = True

        for j in range(M):
            trace_idx = j0 + j - half_M
            if trace_idx < 0 or trace_idx >= nx:
                valid = False
                break

            # 根据道偏移和视速度计算时间延迟。
            offset = (j - half_M) * self.dx
            time_shift = offset / vk
            sample_shift = int(round(time_shift / self.dt))

            for i in range(N):
                sample_idx = i0 + i - half_N + sample_shift
                if sample_idx < 0 or sample_idx >= nt:
                    valid = False
                    break
                w[i, j] = s[sample_idx, trace_idx]

            if not valid:
                break

        return w, valid

    def compute_similarity_coefficient(self, w):
        """
        计算改进的相似系数。

        传统相似系数在各道波形一致但振幅不同时会偏小。
        这里先对每道按振幅归一化，再计算相似度，使其更适合识别相干面波。
        """
        N, M = w.shape

        # 计算每道均方根振幅 Aj。
        Aj = np.zeros(M)
        for j in range(M):
            Aj[j] = np.sqrt(np.sum(w[:, j] ** 2) / N)

        Aj[Aj < 1e-10] = 1e-10

        # 按道振幅归一化。
        wA = np.zeros_like(w)
        for j in range(M):
            wA[:, j] = w[:, j] / Aj[j]

        sum_trace = np.sum(wA, axis=1)
        numerator = np.sum(sum_trace ** 2)
        denominator = M * np.sum(wA ** 2)

        if denominator < 1e-10:
            return 0.0

        Ra = numerator / denominator
        Ra = np.clip(Ra, 0.0, 1.0)
        return Ra

    def determine_optimal_velocity(self, s, i0, j0):
        """
        通过扫描不同视速度，确定当前中心点处的最优视速度。
        """
        Ra_all = np.zeros(self.Nv)

        for k, vk in enumerate(self.v_list):
            w, valid = self.extract_window_data(s, i0, j0, vk)
            if valid:
                Ra_all[k] = self.compute_similarity_coefficient(w)

        k_max = np.argmax(Ra_all)
        v_opt = self.v_list[k_max]
        Ra_max = Ra_all[k_max]

        return v_opt, Ra_max, Ra_all

    def mixed_median_mean_filter(self, U):
        """
        使用混合中值-均值滤波构建初始噪声模型。

        中值滤波可抑制异常振幅，均值滤波可增强沿噪声方向的稳定成分。
        """
        N, M = U.shape
        y = np.zeros(N)

        median_idx = (M + 1) // 2 - 1

        for i in range(N):
            row_sorted = np.sort(U[i, :])

            left_idx = max(0, median_idx - self.p)
            right_idx = min(M - 1, median_idx + self.p)
            neighbors = row_sorted[left_idx:right_idx + 1]

            y[i] = np.mean(neighbors)

        return y

    def weighting_correction(self, y, Ra):
        """
        根据相似系数进行加权修正，避免有效信号被误判为噪声。
        """
        if Ra <= self.c1:
            c = 0.0
        elif Ra >= self.c2:
            c = 1.0
        else:
            c = (Ra - self.c1) / (self.c2 - self.c1)

        yc = y * c
        return yc, c

    def matched_filter(self, yc, s_trace):
        """
        使用简化匹配滤波修正噪声模型的波形和振幅。
        """
        energy_yc = np.sum(yc ** 2)

        if energy_yc < 1e-10:
            return np.zeros_like(yc)

        correlation = np.correlate(s_trace, yc, mode="valid")
        if len(correlation) > 0:
            max_corr_idx = np.argmax(np.abs(correlation))
            corr_val = correlation[max_corr_idx]
        else:
            corr_val = np.sum(s_trace * yc)

        scale = corr_val / energy_yc

        # 限制振幅修正范围，避免局部异常导致估计噪声过度放大。
        scale = np.clip(scale, -2.0, 2.0)

        yf = yc * scale
        return yf

    def process_single_point(self, data, s_low, i0, j0):
        """
        对单个中心点估计面波噪声。
        """
        # 1. 确定最优视速度。
        v_opt, Ra, _ = self.determine_optimal_velocity(s_low, i0, j0)

        # 2. 沿最优视速度提取局部窗口。
        w, valid = self.extract_window_data(s_low, i0, j0, v_opt)
        if not valid:
            return 0.0, v_opt, Ra

        # 3. 构建初始噪声模型。
        y = self.mixed_median_mean_filter(w)

        # 4. 按相似系数加权修正。
        yc, c = self.weighting_correction(y, Ra)

        # 5. 使用中心道原始数据进行匹配滤波。
        half_N = self.N // 2
        trace_data = data[max(0, i0 - half_N):min(data.shape[0], i0 + half_N + 1), j0]

        if len(trace_data) == len(yc):
            yf = self.matched_filter(yc, trace_data)
        else:
            yf = yc * c

        noise_est = yf[half_N] if len(yf) > half_N else yf[len(yf) // 2]
        return noise_est, v_opt, Ra

    def apply(self, data):
        """
        应用 LNF 方法进行相干面波压制。

        Parameters
        ----------
        data : ndarray, shape (nt, nx)
            输入地震数据，维度为时间采样点 x 道号。

        Returns
        -------
        denoised : ndarray
            去噪后的数据。
        noise : ndarray
            估计的面波噪声。
        info : dict
            处理过程中的辅助信息，包括速度场、相似系数字段和低频数据。
        """
        nt, nx = data.shape
        print(f"\n开始 LNF 处理: 数据尺寸 {nt} x {nx}")

        print("步骤 1: 频率分离处理...")
        s_low = self.frequency_division(data)

        noise = np.zeros_like(data)
        v_field = np.zeros_like(data)
        Ra_field = np.zeros_like(data)

        print("步骤 2-5: 逐点估计面波噪声...")
        half_N = self.N // 2
        half_M = self.M // 2

        # 边界处窗口不完整，因此只处理完整窗口覆盖的区域。
        for j0 in range(half_M, nx - half_M):
            if j0 % 10 == 0:
                print(f"  处理进度: {j0}/{nx} 道", end="\r")

            for i0 in range(half_N, nt - half_N):
                noise_est, v_opt, Ra = self.process_single_point(data, s_low, i0, j0)
                noise[i0, j0] = noise_est
                v_field[i0, j0] = v_opt
                Ra_field[i0, j0] = Ra

        print("\n逐点噪声估计完成")

        print("步骤 6: 原始数据减去估计面波噪声...")
        denoised = data - noise

        info = {
            "velocity_field": v_field,
            "similarity_field": Ra_field,
            "low_freq_data": s_low,
        }

        return denoised, noise, info


# ============================================
# 辅助函数：结果可视化
# ============================================

def plot_seismic(data, title, ax=None, cmap="seismic", vmin=None, vmax=None, dt=0.002):
    """绘制地震剖面图。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if vmin is None:
        vmax = np.max(np.abs(data)) * 0.5
        vmin = -vmax

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, data.shape[1], data.shape[0] * dt, 0],
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("道号")
    ax.set_ylabel("时间 (s)")
    return im


def plot_before_after(original, denoised, dt=0.002):
    """绘制去噪前后的地震剖面对比图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vmax = np.max(np.abs(original)) * 0.6

    plot_seismic(original, "原始数据", axes[0], vmax=vmax, dt=dt)
    plot_seismic(denoised, "LNF 去噪后数据", axes[1], vmax=vmax, dt=dt)

    plt.tight_layout()
    return fig


def plot_removed_noise(noise, dt=0.002):
    """绘制 LNF 估计并从原始数据中去除的面波噪声。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    vmax = np.max(np.abs(noise)) * 0.6

    plot_seismic(noise, "LNF 去除的面波噪声", ax, vmax=vmax, dt=dt)

    plt.tight_layout()
    return fig


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    # =========================
    # 参数设置区
    # =========================

    # 数据路径与输出目录
    input_path = Path(r"D:\桌面\面波压制\深层数据面波压制\最终数据\original.npy")
    output_dir = Path.cwd() / "LNF_output"

    # 实际采集参数
    dt = 0.004                  # 时间采样间隔 (s)
    dx = 25                     # 道间距 (m)

    # LNF 处理参数
    lnf_params = {
        "f1": 0,                # 低通滤波通带起始频率 (Hz)
        "f2": 25,               # 低通滤波截止频率 (Hz)
        "v_min": 100,           # 最小扫描视速度 (m/s)
        "v_max": 1200,          # 最大扫描视速度 (m/s)
        "dv": 25,               # 速度扫描间隔 (m/s)
        "win_t": 0.05,          # 时间窗口长度 (s)
        "win_x": 7,             # 空间窗口道数
        "c1": 0.1,              # 加权修正下阈值
        "c2": 0.2,              # 加权修正上阈值
        "p": 2,                 # 混合中值-均值滤波邻域参数
    }

    # 输出文件名
    denoised_filename = "LNF_denoised.npy"
    noise_filename = "LNF_estimated_noise.npy"
    before_after_figure = "LNF_before_after.png"
    removed_noise_figure = "LNF_removed_noise.png"

    # 图片保存参数
    figure_dpi = 150

    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("局部非线性滤波（LNF）- 面波压制")
    print("=" * 60)

    print(f"\n[1] 读取数据: {input_path}")
    data = np.load(input_path)
    data = np.asarray(data, dtype=float).squeeze()
    if data.ndim != 2:
        raise ValueError(f"期望读取二维地震数据，但当前数据维度为 {data.shape}")

    nt, nx = data.shape
    print(f"  数据尺寸: {nt} 个采样点 x {nx} 道")
    print(f"  时间范围: 0-{nt * dt:.3f}s，采样间隔 {dt * 1000:.1f}ms")
    print(f"  道间距: {dx}m")

    print("\n[2] 初始化 LNF 参数...")
    lnf = LocalNonlinearFilter(
        **lnf_params,
        dt=dt,
        dx=dx,
    )

    print("\n[3] 执行 LNF 面波压制...")
    denoised, noise_est, info = lnf.apply(data)

    denoised_path = output_dir / denoised_filename
    noise_path = output_dir / noise_filename
    np.save(denoised_path, denoised)
    np.save(noise_path, noise_est)
    print(f"  已保存去噪数据: {denoised_path}")
    print(f"  已保存估计噪声: {noise_path}")

    print("\n[4] 保存去噪前后对比图...")
    fig = plot_before_after(data, denoised, dt=dt)
    before_after_path = output_dir / before_after_figure
    plt.savefig(before_after_path, dpi=figure_dpi, bbox_inches="tight")
    print(f"  已保存图片: {before_after_path}")

    print("\n[5] 保存去除噪声图...")
    fig_noise = plot_removed_noise(noise_est, dt=dt)
    removed_noise_path = output_dir / removed_noise_figure
    plt.savefig(removed_noise_path, dpi=figure_dpi, bbox_inches="tight")
    print(f"  已保存图片: {removed_noise_path}")

    print("\n" + "=" * 60)
    print(f"LNF 处理完成，结果已保存到: {output_dir}")
    print("=" * 60)

    plt.show()
