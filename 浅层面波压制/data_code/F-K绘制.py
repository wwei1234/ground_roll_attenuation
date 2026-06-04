import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
import os
from pathlib import Path

def get_global_colorbar_range(fk_data_paths, freq_range=None, k_range=None):
    """
    获取所有F-K谱数据的全局色阶范围
    
    参数:
    fk_data_paths: F-K谱数据文件路径列表
    freq_range: 频率显示范围，格式为[fmin, fmax]
    k_range: 波数显示范围，格式为[kmin, kmax]
    
    返回:
    vmin, vmax: 全局最小值和最大值
    """
    global_min = float('inf')
    global_max = float('-inf')
    
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
        
        # 更新全局范围
        current_min = np.min(fk_amp_db)
        current_max = np.max(fk_amp_db)
        
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        
        print(f"文件: {os.path.basename(fk_data_path)}")
        print(f"  范围: [{current_min:.2f}, {current_max:.2f}] dB")
    
    print(f"\n全局色阶范围: [{global_min:.2f}, {global_max:.2f}] dB")
    return global_min, global_max


def plot_standalone_colorbar(vmin, vmax, output_path, 
                             cmap='jet',
                             orientation='horizontal',
                             figsize=(6, 0.6),
                             fontsize=30,
                             tick_direction='in',
                             num_ticks=5,
                             custom_ticks=None,
                             decimal_places=1,
                             label='Amplitude (dB)',
                             dpi=600,
                             save_png=True,
                             save_eps=True,
                             save_pdf=True):
    """
    绘制独立的colorbar
    
    参数:
    vmin, vmax: 色阶范围
    output_path: 输出文件路径（不含扩展名）
    cmap: 色图类型
    orientation: 方向 ('horizontal' 或 'vertical')
    figsize: 图形尺寸 (宽, 高)
    fontsize: 字体大小
    tick_direction: 刻度线方向 ('in', 'out', 或 'inout')
    num_ticks: 刻度数量（当custom_ticks为None时使用）
    custom_ticks: 自定义刻度位置列表，例如 [0, 10, 20, 30, 40]
    decimal_places: 刻度标签小数位数
    label: colorbar标签
    dpi: 图像分辨率
    save_png: 是否保存PNG
    save_eps: 是否保存EPS
    """
    # 设置字体
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 调整子图位置
    if orientation == 'horizontal':
        fig.subplots_adjust(bottom=0.5)
    else:
        fig.subplots_adjust(left=0.5)
    
    # 创建归一化对象
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # 创建colorbar
    if orientation == 'horizontal':
        ticklocation = 'bottom'
    else:
        ticklocation = 'right'
    
    cb = colorbar.ColorbarBase(
        ax, 
        cmap=cmap,
        norm=norm,
        orientation=orientation,
        ticklocation=ticklocation
    )
    
    # 设置刻度位置
    if custom_ticks is not None:
        # 使用自定义刻度
        ticks = np.array(custom_ticks)
        # 确保刻度在色阶范围内
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    else:
        # 使用均匀分布的刻度
        ticks = np.linspace(vmin, vmax, num_ticks)
    
    cb.set_ticks(ticks)
    
    # 设置刻度标签
    tick_labels = [f"{t:.{decimal_places}f}" for t in ticks]
    if orientation == 'horizontal':
        cb.ax.set_xticklabels(tick_labels)
    else:
        cb.ax.set_yticklabels(tick_labels)
    
    # 设置刻度样式
    cb.ax.tick_params(labelsize=fontsize, direction=tick_direction)
    
    # 设置标签（如果需要）
    if label:
        cb.set_label(label, fontsize=fontsize)
    
    # 保存图片
    if save_png:
        png_path = output_path + '.png'
        plt.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Colorbar PNG已保存: {png_path}")
    
    if save_eps:
        eps_path = output_path + '.eps'
        plt.savefig(eps_path, format='eps', dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Colorbar EPS已保存: {eps_path}")
    
    if save_pdf:
        pdf_path = output_path + '.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Colorbar PDF已保存: {pdf_path}")
    
    plt.close(fig)


def plot_fk_spectrum_unified(fk_data_path, output_path, freq_range=None, k_range=None, 
                              fontsize=24, figsize=(6, 5), dpi=300, 
                              vmin=None, vmax=None, save_png=True, save_eps=True,save_pdf=True,
                              show_colorbar=True):
    """
    绘制单个F-K谱（使用统一色阶）
    
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
    show_colorbar: 是否在图上显示colorbar
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
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制F-K谱（使用统一色阶）
    extent_fk = [kx.min(), kx.max(), freqs.max(), freqs.min()]
    im = ax.imshow(fk_amp_db, aspect='auto', cmap='jet',
                   extent=extent_fk, origin='upper',
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    # 设置标签
    ax.set_xlabel('K(1/m)', fontsize=fontsize)
    ax.set_ylabel('Frequency (Hz)', fontsize=fontsize)
    
    # 设置刻度字体大小
    ax.tick_params(axis='both', labelsize=fontsize)
    
    # 添加参考线
    # ax.axhline(y=0, color='w', linestyle='--', linewidth=0.5)
    # ax.axvline(x=0, color='w', linestyle='--', linewidth=0.5)
    
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
    
    # 添加colorbar（可选）
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude (dB)', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
    
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

    if save_pdf:
        pdf_path = output_path + '.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight')
        print(f"PDF已保存: {pdf_path}")
    
    plt.close(fig)


def batch_plot_fk_spectra(fk_data_paths, output_folder, output_names=None,
                          freq_range=None, k_range=None, fontsize=18, 
                          figsize=(8, 7), dpi=600, save_png=True, save_eps=True,save_pdf=True,
                          show_colorbar=False, create_standalone_colorbar=True,
                          colorbar_config=None):
    """
    批量绘制F-K谱（统一色阶）
    
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
    show_colorbar: 是否在F-K谱图上显示colorbar
    create_standalone_colorbar: 是否创建独立的colorbar图片
    colorbar_config: colorbar配置字典，包含以下可选参数:
        - orientation: 'horizontal' 或 'vertical'
        - figsize: (宽, 高)
        - fontsize: 字体大小
        - tick_direction: 'in', 'out', 或 'inout'
        - num_ticks: 刻度数量（当custom_ticks为None时使用）
        - custom_ticks: 自定义刻度位置列表，例如 [0, 10, 20, 30, 40]
        - decimal_places: 小数位数
        - label: 标签文字
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 第一步：获取全局色阶范围
    print("=" * 60)
    print("步骤1: 计算全局色阶范围...")
    print("=" * 60)
    vmin, vmax = get_global_colorbar_range(fk_data_paths, freq_range, k_range)
    
    # 第二步：绘制独立colorbar（如果需要）
    if create_standalone_colorbar:
        print("\n" + "=" * 60)
        print("步骤2: 绘制独立colorbar...")
        print("=" * 60)
        
        # 默认配置
        default_colorbar_config = {
            'orientation': 'horizontal',
            'figsize': (6, 0.6),
            'fontsize': 30,
            'tick_direction': 'in',
            'num_ticks': 5,
            'custom_ticks': None,
            'decimal_places': 1,
            'label': 'Amplitude (dB)'
        }
        
        # 合并用户配置
        if colorbar_config is not None:
            default_colorbar_config.update(colorbar_config)
        
        colorbar_path = os.path.join(output_folder, 'colorbar')
        plot_standalone_colorbar(
            vmin=vmin,
            vmax=vmax,
            output_path=colorbar_path,
            dpi=dpi,
            save_png=save_png,
            save_eps=save_eps,
            save_pdf=save_pdf,
            **default_colorbar_config
        )
    
    # 第三步：批量绘图
    print("\n" + "=" * 60)
    print(f"步骤{'3' if create_standalone_colorbar else '2'}: 批量绘制F-K谱...")
    print("=" * 60)
    
    for i, fk_data_path in enumerate(fk_data_paths):
        # 确定输出文件名
        if output_names is not None and i < len(output_names):
            output_name = output_names[i]
        else:
            # 使用原文件名（去掉扩展名）
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
            save_pdf=save_pdf,
            show_colorbar=show_colorbar
        )
    
    print("\n" + "=" * 60)
    print("批量绘制完成！")
    print(f"所有图片已保存到: {output_folder}")
    print(f"统一色阶范围: [{vmin:.2f}, {vmax:.2f}] dB")
    print("=" * 60)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 文件路径列表
    fk_files = [
        # r'D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\clean_fk.npy',
        # r'D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\DMSSL_noise_fk.npy',
        # r'D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\DMSSL_result_fk.npy',
        # r"D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\F-K_noise_fk.npy",
        # r"D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\F-K_result_fk.npy",
        # r"D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\original_fk.npy",
        # r"D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\U-Net_noise_fk.npy",
        # r"D:\桌面\面波压制\Synthetic data test\F-K谱绘制\F-K数据\U-Net_result_fk.npy",
        # r"D:\桌面\面波压制\Synthetic data test\noise\noise_fk.npy",

        # r"D:\桌面\面波压制\Field data shot65\F-K数据\DMSSL_noise_fk.npy",
        # r"D:\桌面\面波压制\Field data shot65\F-K数据\DMSSL_result_fk.npy",
        # r"D:\桌面\面波压制\Field data shot65\F-K数据\F-K_noise_fk.npy",
        # r"D:\桌面\面波压制\Field data shot65\F-K数据\F-K_result_fk.npy",
        # r"D:\桌面\面波压制\Field data shot65\F-K数据\noisy_fk.npy",
        # r"D:\桌面\面波压制\Field data shot65\F-K数据\U-Net_noise_fk.npy",
        # r"D:\桌面\面波压制\Field data shot65\F-K数据\U-Net_result_fk.npy",


        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\clean_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\DMSSL_noise_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\DMSSL_result_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\F-K_noise_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\F-K_result_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\noised_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\U-Net_noise_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\U-Net_result_fk.npy',
        r'数据汇总\合成记录\源数据汇总_filtered(20-150)_muted_fk\noise_fk.npy',

    ]
    
    # 输出文件夹
    output_folder = r'图片汇总\合成记录filter(20-150)_fk'  # 替换为你的输出文件夹路径
    
    # 自定义colorbar配置
    colorbar_config = {
        'orientation': 'horizontal',  # 或 'vertical'
        'figsize': (6, 0.6),          # 横向: (宽, 高); 竖向可用 (0.6, 6)
        'fontsize': 30,               # 字体大小
        'tick_direction': 'in',       # 'in', 'out', 或 'inout'
        'num_ticks': 5,               # 刻度数量
        'custom_ticks': [-60, -40, -20, 0, 20, 40],
        'decimal_places': 0,          # 小数位数
        'label': ''     # colorbar标签（可设为空字符串''不显示）
    }
    
    # 批量绘制
    batch_plot_fk_spectra(
        fk_data_paths=fk_files,
        output_folder=output_folder,
        output_names=None,
        freq_range=[0, 200],
        k_range=None,
        fontsize=16,
        figsize=(3, 4),
        dpi=600,
        save_png=True,
        save_eps=True,
        save_pdf=True,
        show_colorbar=False,           # 不在F-K谱图上显示colorbar
        create_standalone_colorbar=True,  # 创建独立的colorbar
        colorbar_config=colorbar_config   # colorbar配置
    )