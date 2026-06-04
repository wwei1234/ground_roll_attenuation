from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.ndimage import convolve1d

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore")


@njit(cache=True)
def _extract_window_nb(s_pad, i0, j0, vk, dt, dx, N, M, time_pad):
    half_N = N // 2
    half_M = M // 2
    nr, nc = s_pad.shape
    w = np.zeros((N, M))

    for j in range(M):
        sample_shift = int(round((j - half_M) * dx / (vk * dt)))
        for i in range(N):
            row = i0 + i - half_N + sample_shift + time_pad
            col = j0 + j
            if 0 <= row < nr and 0 <= col < nc:
                w[i, j] = s_pad[row, col]

    return w


@njit(cache=True)
def _similarity_nb(w):
    N, M = w.shape
    amps = np.zeros(M)

    for j in range(M):
        energy = 0.0
        for i in range(N):
            energy += w[i, j] * w[i, j]
        amp = (energy / N) ** 0.5
        amps[j] = amp if amp > 1e-10 else 1e-10

    numerator = 0.0
    denominator = 0.0
    for i in range(N):
        row_sum = 0.0
        for j in range(M):
            val = w[i, j] / amps[j]
            row_sum += val
            denominator += val * val
        numerator += row_sum * row_sum

    denominator *= M
    if denominator < 1e-10:
        return 0.0

    ra = numerator / denominator
    return min(ra, 1.0)


@njit(cache=True)
def _mixed_median_mean_nb(w, p):
    N, M = w.shape
    y = np.zeros(N)
    mid = (M + 1) // 2 - 1

    for i in range(N):
        row = np.sort(w[i, :])
        left = max(0, mid - p)
        right = min(M - 1, mid + p)
        total = 0.0
        count = right - left + 1
        for k in range(left, right + 1):
            total += row[k]
        y[i] = total / count if count > 0 else 0.0

    return y


@njit(cache=True)
def _process_column_nb(j0, data_pad, s_low_pad, v_list, dt, dx, N, M,
                       p, c1, c2, nt, time_pad):
    half_N = N // 2
    half_M = M // 2
    noise_col = np.zeros(nt)
    v_col = np.zeros(nt)
    ra_col = np.zeros(nt)

    for i0 in range(nt):
        best_ra = -1.0
        best_v = v_list[0]

        for k in range(len(v_list)):
            vk = v_list[k]
            w_scan = _extract_window_nb(s_low_pad, i0, j0, vk, dt, dx, N, M, time_pad)
            ra = _similarity_nb(w_scan)
            if ra > best_ra:
                best_ra = ra
                best_v = vk

        w = _extract_window_nb(s_low_pad, i0, j0, best_v, dt, dx, N, M, time_pad)
        y = _mixed_median_mean_nb(w, p)

        if best_ra <= c1:
            weight = 0.0
        elif best_ra >= c2:
            weight = 1.0
        else:
            weight = (best_ra - c1) / (c2 - c1)

        yc = y * weight

        energy = 0.0
        corr = 0.0
        for i in range(N):
            row = i0 + i - half_N + time_pad
            col = j0 + half_M
            energy += yc[i] * yc[i]
            corr += data_pad[row, col] * yc[i]

        if energy > 1e-10:
            scale = corr / energy
            if scale > 2.0:
                scale = 2.0
            if scale < -2.0:
                scale = -2.0
            noise_col[i0] = yc[half_N] * scale

        v_col[i0] = best_v
        ra_col[i0] = best_ra

    return noise_col, v_col, ra_col


def _velocity_list_for_trace(abs_v_list, trace_idx, shot_trace_index, scan_both_directions):
    if shot_trace_index is None:
        if scan_both_directions:
            return np.concatenate((-abs_v_list[::-1], abs_v_list))
        return abs_v_list

    if trace_idx < shot_trace_index:
        return -abs_v_list[::-1]
    if trace_idx > shot_trace_index:
        return abs_v_list
    return np.concatenate((-abs_v_list[::-1], abs_v_list))


class LocalNonlinearFilter:
    def __init__(
        self,
        f1=0,
        f2=25,
        dt=0.002,
        dx=10,
        v_min=100,
        v_max=2000,
        dv=50,
        win_t=0.1,
        win_x=5,
        c1=0.6,
        c2=0.8,
        p=2,
        shot_trace_index=None,
        scan_both_directions=True,
    ):
        self.f1 = f1
        self.f2 = f2
        self.dt = dt
        self.dx = dx
        self.v_min = v_min
        self.v_max = v_max
        self.dv = dv
        self.c1 = c1
        self.c2 = c2
        self.p = p
        self.shot_trace_index = shot_trace_index
        self.scan_both_directions = scan_both_directions

        self.N = int(win_t / dt)
        if self.N % 2 == 0:
            self.N += 1
        self.M = win_x if win_x % 2 == 1 else win_x + 1

        self.abs_v_list = np.arange(v_min, v_max + 0.5 * dv, dv, dtype=np.float64)
        self.max_shift = int(round((self.M // 2) * dx / (v_min * dt)))
        self.time_pad = self.N // 2 + self.max_shift

        self._validate_params()
        self._print_summary()

    def _validate_params(self):
        errors = []
        if not (0 < self.c1 < self.c2 < 1):
            errors.append(f"Need 0 < c1 < c2 < 1, got c1={self.c1}, c2={self.c2}")
        if self.f2 <= self.f1:
            errors.append("Need f2 > f1")
        if self.v_min <= 0 or self.v_max <= self.v_min:
            errors.append("Need 0 < v_min < v_max")
        if self.dv <= 0:
            errors.append("Need dv > 0")
        if errors:
            raise ValueError("\n".join(errors))

        nyquist = 1.0 / (2 * self.dt)
        if self.f2 > nyquist:
            print(f"  [warning] f2={self.f2} Hz is above Nyquist {nyquist:.1f} Hz")
        if self.max_shift >= self.N // 2:
            print(
                f"  [warning] v_min={self.v_min} m/s gives max moveout "
                f"{self.max_shift} samples, larger than half window {self.N // 2}. "
                "Padding is expanded, but a larger win_t or v_min may be more stable."
            )

    def _print_summary(self):
        mode = "shot geometry" if self.shot_trace_index is not None else "both directions"
        print("LNF Plus initialized")
        print(f"  frequency     : {self.f1}-{self.f2} Hz")
        print(f"  speed range   : |v|={self.v_min}-{self.v_max} m/s, dv={self.dv} m/s")
        print(f"  direction     : {mode}, shot_trace_index={self.shot_trace_index}")
        print(f"  window        : {self.N} samples x {self.M} traces")
        print(f"  thresholds    : c1={self.c1}, c2={self.c2}, p={self.p}")

    def _build_lowpass(self, filter_len):
        t = np.arange(filter_len) * self.dt - (filter_len // 2) * self.dt + 1e-10
        df = max(self.f2 - self.f1, 1e-6)
        term1 = 4 * np.sin(np.pi * (self.f2 + self.f1) * t) / (np.pi * t)
        term2 = np.sin(np.pi * df * t / 2) ** 2 / (df ** 2 * np.pi ** 2 * t ** 2)
        h = term1 * term2
        return h / np.sum(np.abs(h))

    def frequency_division(self, data):
        filter_len = min(101, data.shape[0] // 2 * 2 + 1)
        h = self._build_lowpass(filter_len)
        return convolve1d(data.astype(np.float64), h, axis=0, mode="mirror")

    def _pad(self, data):
        half_M = self.M // 2
        return np.pad(data, ((self.time_pad, self.time_pad), (half_M, half_M)), mode="reflect")

    def apply(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")

        nt, nx = data.shape
        print(f"\nStart LNF Plus: {nt} samples x {nx} traces")

        print("Step 1: frequency division")
        s_low = self.frequency_division(data)

        print("Step 2: mirror padding")
        s_low_pad = self._pad(s_low)
        data_pad = self._pad(data)

        print("Step 3: local nonlinear filtering")
        noise = np.zeros_like(data)
        velocity = np.zeros_like(data)
        similarity = np.zeros_like(data)

        t0 = time.time()
        trace_iter = range(nx)
        if HAS_TQDM:
            trace_iter = tqdm(trace_iter, desc="LNF", unit="trace", ncols=80)

        for j0 in trace_iter:
            v_list = _velocity_list_for_trace(
                self.abs_v_list, j0, self.shot_trace_index, self.scan_both_directions
            )
            noise_col, v_col, ra_col = _process_column_nb(
                j0, data_pad, s_low_pad, v_list, self.dt, self.dx,
                self.N, self.M, self.p, self.c1, self.c2, nt, self.time_pad
            )
            noise[:, j0] = noise_col
            velocity[:, j0] = v_col
            similarity[:, j0] = ra_col

            if not HAS_TQDM and (j0 + 1) % 5 == 0:
                elapsed = time.time() - t0
                speed = (j0 + 1) / max(elapsed, 1e-6)
                eta = (nx - j0 - 1) / max(speed, 1e-6)
                print(f"  progress: {j0 + 1}/{nx}, elapsed={elapsed:.0f}s, eta={eta:.0f}s", end="\r")

        print(f"\nNoise estimation finished in {time.time() - t0:.1f} s")
        denoised = data - noise
        info = {
            "low_freq_data": s_low,
            "velocity_field": velocity,
            "similarity_field": similarity,
        }
        return denoised, noise, info


def plot_comparison(original, denoised, noise, dt=0.002, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = np.max(np.abs(original)) * 0.6
    panels = [
        (original, "Original", vmax),
        (denoised, "LNF denoised", vmax),
        (noise, "Estimated noise", vmax * 0.5),
    ]

    for ax, (panel, title, vm) in zip(axes, panels):
        ax.imshow(
            panel,
            aspect="auto",
            cmap="seismic",
            vmin=-vm,
            vmax=vm,
            extent=[0, panel.shape[1], panel.shape[0] * dt, 0],
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time (s)")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    return fig


if __name__ == "__main__":
    input_path = Path(r"D:\桌面\面波压制\深层数据面波压制\最终数据\original.npy")
    output_dir = Path.cwd() / "LNF_output"

    dt = 0.004
    dx = 25

    lnf_params = dict(
        f1=0,
        f2=15,
        v_min=300,
        v_max=650,
        dv=3,
        win_t=0.005,
        win_x=1,
        c1=0.001,
        c2=0.005,
        p=2,
        # Set this to the real shot position in trace-index units.
        # Examples: 0, 64, 63.5, -1 if the shot is left of trace 0.
        # None keeps the old behavior: scan both positive and negative slopes everywhere.
        shot_trace_index=85,
        scan_both_directions=True,
    )
        
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("LNF Plus - ground-roll attenuation")
    print("=" * 60)

    print(f"\n[1] Load data: {input_path}")
    data = np.load(str(input_path)).astype(float).squeeze()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {data.shape}")
    nt, nx = data.shape
    print(f"  data shape: {nt} samples x {nx} traces")
    print(f"  time length: {nt * dt:.3f} s, dt={dt * 1000:.3f} ms, dx={dx} m")

    print("\n[2] Initialize LNF")
    lnf = LocalNonlinearFilter(**lnf_params, dt=dt, dx=dx)

    print("\n[3] Run LNF")
    denoised, noise_est, info = lnf.apply(data)

    print("\n[4] Save results")
    np.save(str(output_dir / "LNF_denoised.npy"), denoised)
    np.save(str(output_dir / "LNF_estimated_noise.npy"), noise_est)
    np.save(str(output_dir / "LNF_velocity_field.npy"), info["velocity_field"])
    np.save(str(output_dir / "LNF_similarity_field.npy"), info["similarity_field"])
    print(f"  output dir: {output_dir}")

    print("\n[5] Save comparison figure")
    plot_comparison(data, denoised, noise_est, dt=dt, save_path=output_dir / "LNF_comparison.png")

    print("\nDone.")
