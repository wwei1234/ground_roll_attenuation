import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from pathlib import Path

def get_global_colorbar_range(fk_data_paths, freq_range=None, k_range=None, 
                               contrast_mode='percentile', percentile_range=(5, 95)):
    """
    获取所有F-K谱数据的全局色阶范围（支持增强对比度）
    
    参数:
    fk_data_paths: F-K谱数据文件路径列表
    freq_range: 频率显示范围，格式为[fmin, fmax]
    k_range: 波数显示范围，格式为[kmin, kmax]
    contrast_mode: 对比度模式
        - 'percentile': 使用百分位数（推荐，可过滤极值）
        - 'std': 使用标准差控制
        - 'minmax': 使用最小最大值（原始方法）
    percentile_range: 百分位数范围，如(5, 95)表示截取5%到95%的数据范围
    
    返回:
    vmin, vmax: 全局最小值和最大值
    """
    all_data = []
    
    for fk_data_path in fk_data_paths:
        # 加载数据
        fk_data = np.load(fk_data_path, allow_pickle=True).item()
        fk_amp_db = fk_data['fk_amp_db']
        freqs = fk_data['freqs']
        kx = fk_data['kx']
        
        # 根据频率和波数范围筛选数据
        if freq_range is not None:
            fmin, fmax = freq_range
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            freq_indices = np.where(freq_mask)[0]
            if len(freq_indices) > 0:
                fk_amp_db = fk_amp_db[freq_indices, :]
        
        if k_range is not None:
            kmin, kmax = k_range
            k_mask = (kx >= kmin) & (kx <= kmax)
            k_indices = np.where(k_mask)[0]
            if len(k_indices) > 0:
                fk_amp_db = fk_amp_db[:, k_indices]
        
        all_data.append(fk_amp_db.flatten())
        
        print(f"文件: {os.path.basename(fk_data_path)}")
        print(f"  原始范围: [{np.min(fk_amp_db):.2f}, {np.max(fk_amp_db):.2f}] dB")
    
    # 合并所有数据
    all_data = np.concatenate(all_data)
    
    # 根据对比度模式计算色阶范围
    if contrast_mode == 'percentile':
        # 使用百分位数，可以过滤掉极端值
        vmin = np.percentile(all_data, percentile_range[0])
        vmax = np.percentile(all_data, percentile_range[1])
        print(f"\n使用百分位数 {percentile_range}: [{vmin:.2f}, {vmax:.2f}] dB")
        
    elif contrast_mode == 'std':
        # 使用均值±N倍标准差
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        n_std = 2.0  # 可调整，越小对比度越强
        vmin = mean_val - n_std * std_val
        vmax = mean_val + n_std * std_val
        print(f"\n使用标准差方法 (±{n_std}σ): [{vmin:.2f}, {vmax:.2f}] dB")
        
    else:  # 'minmax'
        vmin = np.min(all_data)
        vmax = np.max(all_data)
        print(f"\n使用最小最大值: [{vmin:.2f}, {vmax:.2f}] dB")
    
    print(f"全局色阶范围: [{vmin:.2f}, {vmax:.2f}] dB")
    return vmin, vmax


def plot_fk_spectrum_unified(fk_data_path, output_path, freq_range=None, k_range=None, 
                              fontsize=12, figsize=(6, 5), dpi=300, 
                              vmin=None, vmax=None, save_png=True, save_eps=True,
                              gamma_correction=1.0, cmap='jet'):
    """
    绘制单个F-K谱（使用统一色阶，支持伽马校正）
    
    参数:
    fk_data_path: F-K谱数据文件路径
    output_path: 输出文件路径（不含扩展名）
    freq_range: 频率显示范围
    k_range: 波数显示范围
    fontsize: 字体大小
    figsize: 图形尺寸
    dpi: 图像分辨率
    vmin, vmax: 色阶范围
    save_png: 是否保存PNG
    save_eps: 是否保存EPS
    gamma_correction: 伽马校正系数（<1增强弱信号，>1增强强信号）
        - 0.5: 大幅增强对比度
        - 0.7: 中等增强
        - 1.0: 无校正（默认）
    cmap: 色图，可选'jet', 'hot', 'viridis'等
    """
    
    # 设置字体为Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.rm'] = 'Times New Roman'
    rcParams['mathtext.it'] = 'Times New Roman:italic'
    rcParams['mathtext.bf'] = 'Times New Roman:bold'
    
    # 加载F-K谱数据
    fk_data = np.load(fk_data_path, allow_pickle=True).item()
    fk_amp_db = fk_data['fk_amp_db']
    freqs = fk_data['freqs']
    kx = fk_data['kx']
    
    # 应用伽马校正（增强对比度）
    if gamma_correction != 1.0:
        # 归一化到0-1
        fk_normalized = (fk_amp_db - vmin) / (vmax - vmin)
        fk_normalized = np.clip(fk_normalized, 0, 1)
        # 伽马校正
        fk_corrected = np.power(fk_normalized, gamma_correction)
        # 恢复到原始范围
        fk_amp_db = fk_corrected * (vmax - vmin) + vmin
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制F-K谱（使用统一色阶）
    extent_fk = [kx.min(), kx.max(), freqs.max(), freqs.min()]
    im = ax.imshow(fk_amp_db, aspect='auto', cmap=cmap,
                   extent=extent_fk, origin='upper',
                   vmin=vmin, vmax=vmax)
    
    # 设置标签
    ax.set_xlabel('K(1/m)', fontsize=fontsize)
    ax.set_ylabel('Frequency (Hz)', fontsize=fontsize)
    
    # 设置刻度字体大小
    ax.tick_params(axis='both', labelsize=fontsize)
    
    # 添加参考线
    ax.axhline(y=0, color='w', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='w', linestyle='--', linewidth=0.5)
    
    # 设置频率显示范围
    if freq_range is not None:
        fmin, fmax = freq_range
        ax.set_ylim([fmax, fmin])
    else:
        ax.set_ylim([freqs.max(), 0])
    
    # 设置波数显示范围
    if k_range is not None:
        kmin, kmax = k_range
        ax.set_xlim([kmin, kmax])
    else:
        ax.set_xlim([kx.min(), kx.max()])
    
    plt.tight_layout()
    
    # 保存图片
    if save_png:
        png_path = output_path + '.png'
        plt.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"PNG已保存: {png_path}")
    
    if save_eps:
        eps_path = output_path + '.eps'
        plt.savefig(eps_path, format='eps', dpi=dpi, bbox_inches='tight')
        print(f"EPS已保存: {eps_path}")
    
    plt.close(fig)


def batch_plot_fk_spectra(fk_data_paths, output_folder, output_names=None,
                          freq_range=None, k_range=None, fontsize=18, 
                          figsize=(8, 7), dpi=600, save_png=True, save_eps=True,
                          contrast_mode='percentile', percentile_range=(5, 95),
                          gamma_correction=1.0, cmap='jet'):
    """
    批量绘制F-K谱（统一色阶，增强对比度）
    
    参数:
    fk_data_paths: F-K谱数据文件路径列表
    output_folder: 输出文件夹路径
    output_names: 输出文件名列表（不含扩展名），None则使用原文件名
    freq_range: 频率显示范围
    k_range: 波数显示范围
    fontsize: 字体大小
    figsize: 图形尺寸
    dpi: 图像分辨率
    save_png: 是否保存PNG
    save_eps: 是否保存EPS
    contrast_mode: 对比度模式 ('percentile', 'std', 'minmax')
    percentile_range: 百分位数范围，如(5, 95)或(10, 90)
    gamma_correction: 伽马校正系数（推荐0.5-0.7增强对比度）
    cmap: 色图
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 第一步：获取全局色阶范围
    print("=" * 60)
    print("步骤1: 计算全局色阶范围...")
    print(f"对比度模式: {contrast_mode}")
    if contrast_mode == 'percentile':
        print(f"百分位数范围: {percentile_range}")
    if gamma_correction != 1.0:
        print(f"伽马校正: {gamma_correction}")
    print("=" * 60)
    vmin, vmax = get_global_colorbar_range(fk_data_paths, freq_range, k_range,
                                           contrast_mode, percentile_range)
    
    # 第二步：批量绘图
    print("\n" + "=" * 60)
    print("步骤2: 批量绘制F-K谱...")
    print("=" * 60)
    
    for i, fk_data_path in enumerate(fk_data_paths):
        # 确定输出文件名
        if output_names is not None and i < len(output_names):
            output_name = output_names[i]
        else:
            output_name = Path(fk_data_path).stem
        
        output_path = os.path.join(output_folder, output_name)
        
        print(f"\n[{i+1}/{len(fk_data_paths)}] 绘制: {os.path.basename(fk_data_path)}")
        
        # 绘制F-K谱
        plot_fk_spectrum_unified(
            fk_data_path=fk_data_path,
            output_path=output_path,
            freq_range=freq_range,
            k_range=k_range,
            fontsize=fontsize,
            figsize=figsize,
            dpi=dpi,
            vmin=vmin,
            vmax=vmax,
            save_png=save_png,
            save_eps=save_eps,
            gamma_correction=gamma_correction,
            cmap=cmap
        )
    
    print("\n" + "=" * 60)
    print("批量绘制完成！")
    print(f"所有图片已保存到: {output_folder}")
    print(f"统一色阶范围: [{vmin:.2f}, {vmax:.2f}] dB")
    print("=" * 60)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    fk_files = [
        # r'D:\桌面\面波压制\实际数据结果\F-K谱\DMSSL2_noise_fk.npy',
        # r'D:\桌面\面波压制\实际数据结果\F-K谱\DMSSL2_result_fk.npy',
        # r'D:\桌面\面波压制\实际数据结果\F-K谱\F-K_noise_fk.npy',
        # r"D:\桌面\面波压制\实际数据结果\F-K谱\F-K_result_fk.npy",
        # r"D:\桌面\面波压制\实际数据结果\F-K谱\Original_fk.npy",
        # r"D:\桌面\面波压制\实际数据结果\F-K谱\U-Net_noise_fk.npy",
        # r"D:\桌面\面波压制\实际数据结果\F-K谱\U-Net_result_fk.npy",

        r"D:\桌面\面波压制\Field data shot50\F-K数据\DMSSL_noise_fk.npy"
        r"D:\桌面\面波压制\Field data shot50\F-K数据\DMSSL_result_fk.npy"
        r"D:\桌面\面波压制\Field data shot50\F-K数据\F-K_noise_fk.npy"
        r"D:\桌面\面波压制\Field data shot50\F-K数据\F-K_result_fk.npy"
        r"D:\桌面\面波压制\Field data shot50\F-K数据\original_fk.npy"
        r"D:\桌面\面波压制\Field data shot50\F-K数据\U-Net_noise_fk.npy"
        r"D:\桌面\面波压制\Field data shot50\F-K数据\U-Net_result_fk.npy"
    ]
    
    output_folder = r'D:\桌面\面波压制\Field data shot50\F-K图'
    
    # ===== 推荐配置（选择一种）=====
    
    # # 配置1：使用百分位数 + 伽马校正（推荐，效果最好）
    # batch_plot_fk_spectra(
    #     fk_data_paths=fk_files,
    #     output_folder=output_folder,
    #     output_names=None,
    #     freq_range=[0, 200],
    #     k_range=None,
    #     fontsize=18,
    #     figsize=(8, 7),
    #     dpi=600,
    #     save_png=True,
    #     save_eps=True,
    #     contrast_mode='percentile',  # 使用百分位数
    #     percentile_range=(5, 95),    # 截取5%-95%的数据范围（可调整为(10,90)更强烈）
    #     gamma_correction=0.6,        # 伽马校正，0.5-0.7增强对比度（越小越强烈）
    #     cmap='jet'                   # 也可试试'hot', 'viridis'
    # )
    
    # 配置2：仅使用百分位数（温和增强）
    batch_plot_fk_spectra(
        fk_data_paths=fk_files,
        output_folder=output_folder,
        freq_range=[0, 200],
        contrast_mode='percentile',
        percentile_range=(10, 90),  # 更窄的范围
        gamma_correction=1.0,       # 不使用伽马校正
    )
    
    # 配置3：仅使用伽马校正（保留完整动态范围）
    # batch_plot_fk_spectra(
    #     fk_data_paths=fk_files,
    #     output_folder=output_folder,
    #     freq_range=[0, 500],
    #     contrast_mode='minmax',     # 使用完整范围
    #     gamma_correction=0.5,       # 强烈增强对比度
    # )

    