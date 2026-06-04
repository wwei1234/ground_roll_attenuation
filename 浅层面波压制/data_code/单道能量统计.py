import numpy as np
import matplotlib.pyplot as plt
import segyio
from matplotlib import rcParams

# 设置字体为Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['axes.unicode_minus'] = False

# ============================================================
# 全局字体大小设置（在此处统一修改）
LABEL_FONTSIZE  = 24   # 坐标轴标题字体大小
TICK_FONTSIZE   = 24   # 刻度字体大小
LEGEND_FONTSIZE = 12   # 图例字体大小

# 全局图尺寸设置（宽, 高），单位英寸
FIG_WIDTH  = 12        # 图宽
FIG_HEIGHT = 4         # 图高
# ============================================================

class SeismicEnergyAnalyzer:
    """地震数据能量片段分析器"""
    
    def __init__(self, window_width=0.08, time_shift=0.01):
        self.window_width = window_width
        self.time_shift = time_shift
        
    def gaussian_window(self, t, center, alpha):
        X = np.sqrt(alpha / np.pi)
        window = X * np.exp(-alpha**2 * (t - center)**2)
        return window
    
    def compute_segment_energy(self, seismic_trace, dt):
        n_samples = len(seismic_trace)
        t = np.arange(n_samples) * dt
        alpha = 1.0 / self.window_width
        n_windows = int((t[-1] - t[0]) / self.time_shift) + 1
        time_centers = np.arange(n_windows) * self.time_shift + t[0]
        
        energies = []
        segments = []
        for center in time_centers:
            window = self.gaussian_window(t, center, alpha)
            segment = seismic_trace * window
            segments.append(segment)
            avg_amplitude = np.mean(np.abs(segment))
            energies.append(avg_amplitude)
        
        return time_centers, np.array(energies), segments
    
    def identify_surface_wave(self, energies, threshold=0.001):
        return energies > threshold

    def _apply_ax_style(self, ax):
        """统一应用坐标轴样式：无网格、刻度线朝内（四边）"""
        ax.grid(False)
        ax.tick_params(direction='in', labelsize=TICK_FONTSIZE,
                       top=True, right=True)

    def plot_three_figures_separate(self, seismic_trace, dt, time_centers, energies,
                          threshold=0.001, window_index=None, interactive=False,
                          save_figures=False, save_dir='./'):
        t = np.arange(len(seismic_trace)) * dt
        alpha = 1.0 / self.window_width
        center_time = time_centers[window_index]
        gaussian_win = self.gaussian_window(t, center_time, alpha)
        segment = seismic_trace * gaussian_win
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        
        # ========== 第一张图：原始地震道 + 高斯时窗 ==========
        fig1 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax1 = fig1.add_subplot(111)
        ax1.plot(t, seismic_trace, 'k-', linewidth=1.5, label='Raw seismic trace')
        gaussian_win_scaled = gaussian_win * np.max(np.abs(seismic_trace)) / np.max(gaussian_win)
        ax1.plot(t, gaussian_win_scaled, 'k--', linewidth=1.5,
                label=f'Gaussian window (center={center_time:.2f}s)')
        if np.any(surface_wave_mask):
            for sw_time in time_centers[surface_wave_mask]:
                ax1.axvspan(sw_time - self.window_width/2,
                          sw_time + self.window_width/2,
                          alpha=0.2, color='gray')
        ax1.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Amplitude', fontsize=LABEL_FONTSIZE)
        ax1.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')
        ax1.set_xlim([t[0], t[-1]])
        self._apply_ax_style(ax1)
        plt.tight_layout()
        
        # ========== 第二张图：地震道片段 ==========
        fig2 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax2 = fig2.add_subplot(111)
        ax2.plot(t, segment, 'k-', linewidth=1.5, label='Seismic trace segment')
        ax2.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel('Amplitude', fontsize=LABEL_FONTSIZE)
        ax2.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')
        ax2.set_xlim([t[0], t[-1]])
        self._apply_ax_style(ax2)
        plt.tight_layout()
        
        # ========== 第三张图：能量统计图 ==========
        fig3 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax3 = fig3.add_subplot(111)
        ax3.plot(time_centers, energies, 'k-', linewidth=2, label='Segment amplitude')
        ax3.axhline(y=threshold, color='k', linestyle='--',
                   linewidth=1.5, label=f'Ground roll threshold = {threshold}')
        if np.any(surface_wave_mask):
            ax3.fill_between(time_centers, 0, energies,
                           where=surface_wave_mask,
                           alpha=0.3, color='gray',
                           label='Identified ground-roll region')
        ax3.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
        ax3.set_ylabel('Amplitude', fontsize=LABEL_FONTSIZE)
        ax3.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')
        ax3.set_xlim([time_centers[0], time_centers[-1]])
        self._apply_ax_style(ax3)
        plt.tight_layout()
        
        # 保存图片
        if save_figures:
            import os
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f"\n正在保存图片到目录: {save_dir}")
            for fig, name in [(fig1, 'figure1_trace_and_window'),
                              (fig2, 'figure2_segment'),
                              (fig3, 'figure3_energy')]:
                fig.savefig(os.path.join(save_dir, f'{name}.png'), dpi=600, bbox_inches='tight')
                fig.savefig(os.path.join(save_dir, f'{name}.pdf'), bbox_inches='tight')
                fig.savefig(os.path.join(save_dir, f'{name}.eps'), bbox_inches='tight')
                print(f"  ✓ {name} 保存完成 (PNG, PDF, EPS)")
            print(f"\n所有图片已保存!")
        
        return fig1, fig2, fig3, window_index
    
    def plot_energy_statistics(self, time_centers, energies, threshold=0.001,
                              title='Seismic trace segment energy statistics'):
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax = fig.add_subplot(111)
        ax.plot(time_centers, energies, 'b-', linewidth=2, label='Segment amplitude')
        ax.axhline(y=threshold, color='r', linestyle='--',
                   linewidth=1.5, label=f'Surface wave threshold = {threshold}')
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        if np.any(surface_wave_mask):
            ax.fill_between(time_centers, 0, energies,
                           where=surface_wave_mask,
                           alpha=0.3, color='gray',
                           label='Identified surface wave region')
        ax.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Amplitude', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.set_xlim([time_centers[0], time_centers[-1]])
        self._apply_ax_style(ax)
        plt.tight_layout()
        return fig
    
    def plot_trace_with_windows(self, seismic_trace, dt, time_centers,
                               energies, threshold=0.001):
        t = np.arange(len(seismic_trace)) * dt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
        
        ax1.plot(t, seismic_trace, 'k-', linewidth=1)
        ax1.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Amplitude', fontsize=LABEL_FONTSIZE)
        ax1.set_xlim([t[0], t[-1]])
        self._apply_ax_style(ax1)
        
        surface_wave_mask = self.identify_surface_wave(energies, threshold)
        if np.any(surface_wave_mask):
            for tw_time in time_centers[surface_wave_mask]:
                ax1.axvspan(tw_time - self.window_width/2,
                          tw_time + self.window_width/2,
                          alpha=0.2, color='gray')
        
        ax2.plot(time_centers, energies, 'b-', linewidth=2, label='Segment amplitude')
        ax2.axhline(y=threshold, color='r', linestyle='--',
                   linewidth=1.5, label=f'Threshold = {threshold}')
        ax2.fill_between(time_centers, 0, energies,
                        where=surface_wave_mask,
                        alpha=0.3, color='gray',
                        label='Surface wave region')
        ax2.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel('Amplitude', fontsize=LABEL_FONTSIZE)
        ax2.legend(fontsize=LEGEND_FONTSIZE)
        ax2.set_xlim([time_centers[0], time_centers[-1]])
        self._apply_ax_style(ax2)
        
        plt.tight_layout()
        return fig
    
def read_segy(data_dir, shotnum=0):
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)
        shot_num = shotnum if shotnum else len(set(sourceX))
        len_shot = trace_num // shot_num
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num, time, len_shot))
        for j in range(shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data


if __name__ == "__main__":
    dt = 0.00025
    seismic_trace = read_segy(r"训练数据\dataset4_700.sgy", shotnum=700)[0][:, 32]
    print(seismic_trace.shape)
    seismic_trace = seismic_trace / np.max(np.abs(seismic_trace))

    analyzer = SeismicEnergyAnalyzer(window_width=0.01, time_shift=0.002)
    time_centers, energies, segments = analyzer.compute_segment_energy(seismic_trace, dt)

    threshold = 0.08
    surface_wave_mask = analyzer.identify_surface_wave(energies, threshold)
    if np.any(surface_wave_mask):
        sw_times = time_centers[surface_wave_mask]
        print(f"识别的面波时间范围: {sw_times[0]:.2f}s ~ {sw_times[-1]:.2f}s")

    fig1, fig2, fig3, selected_idx = analyzer.plot_three_figures_separate(
        seismic_trace, dt, time_centers, energies, threshold,
        window_index=50,
        save_figures=True,
        save_dir=r'D:\桌面\面波压制\浅层面波压制\图片汇总\矢量图'
    )
    print(f"\n显示的窗口索引: {selected_idx}, 中心时间: {time_centers[selected_idx]:.3f}s")
    plt.show()