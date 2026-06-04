import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==================== Matplotlib 中文字体设置 ====================
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


# ==================== 直达波掩码函数 ====================
def get_direct_wave_mask(time_samples, trace_num, center_trace,
                        velocity, mute_time0, dx, dt):
    """计算直达波区域掩码"""
    direct_mask = np.zeros((time_samples, trace_num), dtype=bool)
    
    for tr in range(trace_num):
        offset = abs(tr - center_trace) * dx
        t0 = mute_time0 + offset / velocity
        t0_idx = int(t0 / dt)
        if t0_idx < time_samples:
            direct_mask[t0_idx:, tr] = True
    
    return direct_mask


# ==================== 绘制地震波形图函数 (增强弱信号显示) ====================
def plot_seismic_wiggle(ax, data, dt, vmax_global, fill_color='black',
                        center_trace=None, trace_skip=1, gain=1.0,
                        normalize_method='global'):
    """
    绘制地震炮集波形图（Variable Area / Wiggle Trace）。
    正振幅区域填充指定颜色。
    
    参数:
        ax: Matplotlib axes 对象
        data: (time_samples, trace_num) 形状的地震数据
        dt: 时间采样间隔
        vmax_global: 全局最大振幅，仅在 normalize_method='global' 时使用
        fill_color: 正振幅填充颜色
        center_trace: 炮点道号，用于绘制指示线
        trace_skip: 绘制时跳过的道数，用于稀疏显示
        gain: 迹增益因子，控制波形幅度
        normalize_method: 归一化方法 ('global' 或 'trace')
    """
    time_samples, trace_num = data.shape
    times = np.arange(time_samples) * dt
    
    # 设置绘制范围
    ax.set_xlim(0, trace_num)
    ax.set_ylim(times[-1], times[0])
    
    # 每个迹的最大横向跨度（归一化到半个迹距）
    trace_x_max = 0.5 * gain 
    
    for i in range(0, trace_num, trace_skip):
        trace = data[:, i].copy()
        
        # 归一化和缩放逻辑
        if normalize_method == 'trace':
            # 迹间归一化 (Trace-by-trace normalization): 增强弱反射波
            vmax_trace = np.nanmax(np.abs(trace))
            if vmax_trace > 1e-9: # 避免除以零
                trace /= vmax_trace
            # x_val: 迹的波形横坐标值，相对于迹中心 i
            x_val = i + trace * trace_x_max 
        elif normalize_method == 'global':
            # 全局归一化 (Global normalization): 保持相对振幅关系
            # x_val: 迹的波形横坐标值，相对于迹中心 i
            x_val = i + trace * trace_x_max / vmax_global
        else:
             raise ValueError("normalize_method 必须是 'global' 或 'trace'")
        
        # 绘制迹（波形）
        ax.plot(x_val, times, color='black', linewidth=0.5)
        
        # 填充正振幅区域
        positive_samples = trace > 0
        
        # 构造填充的多边形顶点，只在正振幅区域进行填充
        x_fill = np.where(positive_samples, x_val, i)

        # 使用 fill_between 简化填充操作：
        ax.fill_betweenx(times, i, x_fill, where=(x_fill > i), 
                         facecolor=fill_color, edgecolor='none', interpolate=True)
        
    ax.set_facecolor('white') # 设置背景色为白色
    ax.tick_params(axis='y', labelleft=True)
    
    # 绘制炮点线
    if center_trace is not None:
        ax.axvline(center_trace, color='lime', linestyle='--', linewidth=2, label='炮点')


# ==================== 对比绘制函数 (修改后 - 启用 Trace 归一化) ====================
def plot_comparison(original, network_input, denoised, sw_mask,
                    shot_idx, shot_center=22,
                    direct_velocity=800, mute_time0=0.010, dx=2, dt=0.00025,
                    input_with_mask=True, boost_enabled=False,
                    energy_boost_factor=1.0,
                    save_path=None, figsize=(24, 16)):

    # 布局: 2行, 4列，实际使用7个
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    time_samples, trace_num = original.shape

    # 计算直达波掩码
    direct_wave_mask = get_direct_wave_mask(
        time_samples, trace_num, shot_center,
        direct_velocity, mute_time0, dx, dt
    )

    # vmax 用于全局归一化
    vmax_global = np.nanpercentile(np.abs(original), 99)
    # 波形增益因子。迹间归一化后，gain=0.8 表示波形最大宽度为 0.8 倍迹间距。
    wiggle_gain_strong = 0.8 
    wiggle_gain_diff = 1.6 # 残差图可能需要更高的增益

    # ==================== 第一行 - 4张图 ====================
    
    # 1. 原始数据 (axes[0, 0]) -> **使用 Trace 归一化**
    plot_seismic_wiggle(axes[0, 0], original, dt, vmax_global, 
                        center_trace=shot_center, gain=wiggle_gain_strong, 
                        normalize_method='trace')
    axes[0, 0].set_title(f'预处理后原始数据（波形，迹间归一化） - 第{shot_idx}炮', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('道号')
    axes[0, 0].set_ylabel('时间 (s)')
    axes[0, 0].legend(loc='upper right', fontsize=8)

    # 2. 面波区域标识 (axes[0, 1]) -> 保持色度图 (imshow)
    sw_mask_display = sw_mask.copy()
    sw_mask_display[~direct_wave_mask] = 0
    im1 = axes[0, 1].imshow(sw_mask_display, aspect='auto', cmap='RdYlGn_r',
                            vmin=0, vmax=1, interpolation='nearest')
    axes[0, 1].set_title('面波区域标识', fontsize=11)
    axes[0, 1].set_xlabel('道号')
    axes[0, 1].set_ylabel('时间采样点')
    axes[0, 1].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
    plt.colorbar(im1, ax=axes[0, 1])

    # 3. 网络输入 (axes[0, 2]) -> **使用 Trace 归一化**
    plot_seismic_wiggle(axes[0, 2], network_input, dt, vmax_global, 
                        center_trace=shot_center, gain=wiggle_gain_strong,
                        normalize_method='trace')
    title_str = f'网络输入（波形，迹间归一化，面波→掩码）' if input_with_mask else '网络输入（波形，迹间归一化，原始）'
    axes[0, 2].set_title(title_str, fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel('道号')
    axes[0, 2].set_ylabel('时间 (s)')

    # 4. 网络输入 + 面波区域叠加 (axes[0, 3]) -> 保持色度图（仅叠加）
    axes[0, 3].imshow(network_input, aspect='auto', cmap='gray',
                      vmin=-vmax_global, vmax=vmax_global, interpolation='bilinear')
    masked_sw = np.ma.masked_where(sw_mask_display == 0, sw_mask_display)
    axes[0, 3].imshow(masked_sw, aspect='auto', cmap='Reds',
                      alpha=0.4, vmin=0, vmax=1, interpolation='nearest')
    axes[0, 3].set_title('网络输入（灰度） + 面波区域叠加', fontsize=11)
    axes[0, 3].set_xlabel('道号')
    axes[0, 3].set_ylabel('时间采样点')
    axes[0, 3].axvline(shot_center, color='lime', linestyle='--', linewidth=2)

    
    # ==================== 第二行 - 3张图 ====================
    
    # 5. 最终结果 (axes[1, 0]) -> **使用 Trace 归一化**
    plot_seismic_wiggle(axes[1, 0], denoised, dt, vmax_global, 
                        center_trace=shot_center, gain=wiggle_gain_strong,
                        normalize_method='trace')
    result_title = (f'最终结果（波形，迹间归一化，增强×{energy_boost_factor:.2f}+拼接）'
                    if boost_enabled else '最终结果（波形，迹间归一化，直接拼接）')
    axes[1, 0].set_title(result_title, fontsize=11, fontweight='bold', color='green')
    axes[1, 0].set_xlabel('道号')
    axes[1, 0].set_ylabel('时间 (s)')
    
    # 6. 面波区域的去噪效果 (axes[1, 1]) -> 使用全局归一化和高增益
    sw_change = np.zeros_like(original)
    valid_sw = sw_mask & direct_wave_mask
    sw_change[valid_sw] = original[valid_sw] - denoised[valid_sw]
    vmax_sw_change = np.percentile(np.abs(sw_change[valid_sw]), 99) if valid_sw.any() else 1.0
    
    plot_seismic_wiggle(axes[1, 1], sw_change, dt, vmax_sw_change, 
                        center_trace=shot_center, gain=wiggle_gain_diff,
                        normalize_method='global')
    axes[1, 1].set_title('面波去噪残差（波形，原始 - 去噪）', fontsize=11)
    axes[1, 1].set_xlabel('道号')
    axes[1, 1].set_ylabel('时间 (s)')

    # 7. 总体差异 (axes[1, 2]) -> 使用全局归一化和高增益
    total_diff = original - denoised
    vmax_diff = np.percentile(np.abs(total_diff), 99)
    
    plot_seismic_wiggle(axes[1, 2], total_diff, dt, vmax_diff, 
                        center_trace=shot_center, gain=wiggle_gain_diff,
                        normalize_method='global')
    axes[1, 2].set_title('总体差异（波形，原始 - 最终）', fontsize=11)
    axes[1, 2].set_xlabel('道号')
    axes[1, 2].set_ylabel('时间 (s)')

    # 移除未使用的子图 axes[1, 3]
    fig.delaxes(axes[1, 3]) 

    # 美化排版
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(wspace=0.3, hspace=0.3) 

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {save_path}")
    plt.show()


# ==================== 数据加载与绘图调用 ====================
# 数据路径保持不变，假设这些文件存在
original = np.load(r"D:\桌面\面波压制\Synthetic data test2\DMSSL\original_shot_0.npy")
network_input = np.load(r"D:\桌面\面波压制\Synthetic data test2\DMSSL\input_shot_0.npy")
denoised = np.load(r"D:\桌面\面波压制\Synthetic data test2\DMSSL\denoised_shot_0.npy")
sw_mask = np.load(r"D:\桌面\面波压制\Synthetic data test2\DMSSL\sw_mask_shot_0.npy")


# 调用修改后的函数
plot_comparison(original, network_input, denoised, sw_mask,
                shot_idx=100, shot_center=24,
                direct_velocity=800, mute_time0=0.010, dx=2, dt=0.00025,
                input_with_mask=True, boost_enabled=True,
                energy_boost_factor=1.5,
                save_path=r"D:\桌面\面波压制\自监督高斯掩码\denoised_results_energy_boost\comparison_shot_100_7panels_wiggle_enhanced.png", 
                figsize=(24, 16))