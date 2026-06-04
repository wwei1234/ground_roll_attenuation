import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def visualize_npy_seismic_clean(npy_path, save_prefix=None, dpi=600, 
                                 figsize=(10, 18), cmap='seismic',
                                 use_percentile=True, percentile=99,
                                 vmin=None, vmax=None,
                                 force_symmetric=True):
    """
    绘制单个npy文件（地震数据格式）- 与原代码显示效果一致
    
    参数:
        npy_path: npy文件路径
        save_prefix: 保存路径前缀（如果为None则只显示不保存）
        dpi: 图像分辨率
        figsize: 图像尺寸 (宽, 高)
        cmap: 颜色映射 ('seismic', 'gray', 'jet', 等)
        use_percentile: 是否使用百分位数自动调整显示范围
        percentile: 百分位数值（当use_percentile=True时使用）
        vmin, vmax: 手动指定显示范围（当use_percentile=False时使用）
        force_symmetric: 强制使用对称的显示范围 (-vmax, vmax)
    
    返回:
        fig: matplotlib figure对象
    """
    # 加载数据
    data = np.load(npy_path)
    print(f"已加载数据: {npy_path}")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
    
    # 确定显示范围（与原代码保持一致）
    if use_percentile:
        # 使用绝对值的百分位数，确保对称显示
        vmax_calc = np.percentile(np.abs(data), percentile)
        vmin_calc = -vmax_calc
        print(f"使用{percentile}百分位数: vmin={vmin_calc:.6f}, vmax={vmax_calc:.6f}")
    else:
        if vmin is None or vmax is None:
            if force_symmetric:
                vmax_calc = np.max(np.abs(data))
                vmin_calc = -vmax_calc
            else:
                vmin_calc = np.min(data)
                vmax_calc = np.max(data)
        else:
            vmin_calc = vmin
            vmax_calc = vmax
        print(f"使用指定范围: vmin={vmin_calc:.6f}, vmax={vmax_calc:.6f}")
    
    # 创建图像
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # 绘制数据（使用与原代码相同的插值方法）
    ax.imshow(
        data, 
        aspect='auto', 
        cmap=cmap, 
        vmin=vmin_calc, 
        vmax=vmax_calc, 
        interpolation='bilinear'
    )
    
    # 去除所有装饰
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 保存图像
    if save_prefix:
        output_path = f'{save_prefix}.png'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"已保存图像: {output_path}")
    
    return fig


def visualize_multiple_npy_clean(npy_paths, save_prefix=None, dpi=600,
                                  figsize=(10, 18), cmaps=None,
                                  use_percentile=True, percentile=99,
                                  titles=None, force_symmetric=True,
                                  use_unified_vmax=True, 
                                  reference_idx=0):
    """
    批量绘制多个npy文件 - 与原代码显示效果一致
    
    参数:
        npy_paths: npy文件路径列表
        save_prefix: 保存路径前缀（如果为None则只显示不保存）
        dpi: 图像分辨率
        figsize: 单个图像尺寸 (宽, 高)
        cmaps: 颜色映射列表（如果为None则全部使用'seismic'）
        use_percentile: 是否使用百分位数自动调整显示范围
        percentile: 百分位数值
        titles: 标题列表（可选，用于打印信息）
        force_symmetric: 强制使用对称的显示范围
        use_unified_vmax: 是否使用统一的vmax（基于参考数据）
        reference_idx: 参考数据的索引（通常是原始数据，索引0）
    
    返回:
        fig_list: matplotlib figure对象列表
    """
    if cmaps is None:
        cmaps = ['seismic'] * len(npy_paths)
    
    if titles is None:
        titles = [f"图{i+1}" for i in range(len(npy_paths))]
    
    # 如果使用统一vmax，先计算参考数据的vmax
    unified_vmax = None
    unified_vmin = None
    if use_unified_vmax and len(npy_paths) > 0:
        reference_data = np.load(npy_paths[reference_idx])
        if use_percentile:
            unified_vmax = np.percentile(np.abs(reference_data), percentile)
        else:
            unified_vmax = np.max(np.abs(reference_data))
        unified_vmin = -unified_vmax
        print(f"\n{'='*60}")
        print(f"使用统一显示范围（基于{titles[reference_idx]}）:")
        print(f"  vmin = {unified_vmin:.6f}")
        print(f"  vmax = {unified_vmax:.6f}")
        print(f"{'='*60}")
    
    fig_list = []
    
    for idx, (npy_path, cmap, title) in enumerate(zip(npy_paths, cmaps, titles)):
        print(f"\n{'='*60}")
        print(f"正在处理 {title}: {npy_path}")
        print(f"{'='*60}")
        
        # 构建保存路径
        if save_prefix:
            current_save_prefix = f'{save_prefix}_{idx+1}_{title}'
        else:
            current_save_prefix = None
        
        # 绘制单个图像
        if use_unified_vmax:
            fig = visualize_npy_seismic_clean(
                npy_path=npy_path,
                save_prefix=current_save_prefix,
                dpi=dpi,
                figsize=figsize,
                cmap=cmap,
                use_percentile=False,
                vmin=unified_vmin,
                vmax=unified_vmax,
                force_symmetric=force_symmetric
            )
        else:
            fig = visualize_npy_seismic_clean(
                npy_path=npy_path,
                save_prefix=current_save_prefix,
                dpi=dpi,
                figsize=figsize,
                cmap=cmap,
                use_percentile=use_percentile,
                percentile=percentile,
                force_symmetric=force_symmetric
            )
        
        fig_list.append(fig)
    
    return fig_list


def visualize_mask_npy_clean(mask_path, save_prefix=None, dpi=600,
                              figsize=(10, 18), cmap='bwr'):
    """
    专门用于绘制掩码数据（布尔型或0-1范围）
    
    参数:
        mask_path: 掩码npy文件路径
        save_prefix: 保存路径前缀
        dpi: 图像分辨率
        figsize: 图像尺寸 (宽, 高)
        cmap: 颜色映射（'bwr'=蓝白红, 'RdYlGn_r'=红黄绿反转）
    
    返回:
        fig: matplotlib figure对象
    """
    # 加载数据
    mask_data = np.load(mask_path)
    print(f"已加载掩码数据: {mask_path}")
    print(f"数据形状: {mask_data.shape}")
    print(f"数据类型: {mask_data.dtype}")
    print(f"唯一值: {np.unique(mask_data)}")
    
    # 创建图像
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # 绘制掩码（使用与原代码相同的插值方法）
    ax.imshow(
        mask_data, 
        aspect='auto', 
        cmap=cmap, 
        vmin=0, 
        vmax=1, 
        interpolation='nearest'
    )
    
    # 去除所有装饰
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 保存图像
    if save_prefix:
        output_path = f'{save_prefix}.png'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"已保存掩码图像: {output_path}")
    
    return fig


def visualize_rgb_mask_clean(sw_mask_path, non_sw_mask_path, 
                              save_prefix=None, dpi=600, figsize=(10, 18)):
    """
    绘制面波(红) + 非面波(绿) 组合掩码图
    
    参数:
        sw_mask_path: 面波掩码npy文件路径
        non_sw_mask_path: 非面波掩码npy文件路径
        save_prefix: 保存路径前缀
        dpi: 图像分辨率
        figsize: 图像尺寸 (宽, 高)
    
    返回:
        fig: matplotlib figure对象
    """
    # 加载数据
    sw_mask = np.load(sw_mask_path).astype(bool)
    non_sw_mask = np.load(non_sw_mask_path).astype(bool)
    
    print(f"已加载面波掩码: {sw_mask_path}")
    print(f"已加载非面波掩码: {non_sw_mask_path}")
    
    time_samples, trace_num = sw_mask.shape
    
    # 创建RGB图像
    mask_rgb = np.zeros((time_samples, trace_num, 3))
    mask_rgb[:, :, 2] = 1.0  # 蓝色背景
    mask_rgb[sw_mask] = [1.0, 0.0, 0.0]  # 红色面波
    mask_rgb[non_sw_mask] = [0.0, 1.0, 0.0]  # 绿色非面波
    
    # 创建图像
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.imshow(mask_rgb, aspect='auto', interpolation='nearest')
    
    # 去除所有装饰
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 保存图像
    if save_prefix:
        output_path = f'{save_prefix}.png'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"已保存RGB掩码图像: {output_path}")
    
    return fig


def visualize_mask_values_clean(original_data_path, input_data_path,
                                 sw_mask_path, non_sw_mask_path,
                                 save_prefix=None, dpi=600, figsize=(10, 18),
                                 percentile=99):
    """
    绘制掩码值图（仅显示被掩码位置的值）
    
    参数:
        original_data_path: 原始数据npy文件路径
        input_data_path: 输入数据npy文件路径
        sw_mask_path: 面波掩码npy文件路径
        non_sw_mask_path: 非面波掩码npy文件路径
        save_prefix: 保存路径前缀
        dpi: 图像分辨率
        figsize: 图像尺寸 (宽, 高)
        percentile: 百分位数
    
    返回:
        fig: matplotlib figure对象
    """
    # 加载数据
    original_data = np.load(original_data_path)
    input_data = np.load(input_data_path)
    sw_mask = np.load(sw_mask_path).astype(bool)
    non_sw_mask = np.load(non_sw_mask_path).astype(bool)
    
    print(f"已加载原始数据: {original_data_path}")
    print(f"已加载输入数据: {input_data_path}")
    print(f"已加载面波掩码: {sw_mask_path}")
    print(f"已加载非面波掩码: {non_sw_mask_path}")
    
    # 创建掩码值图
    mask_values = np.zeros_like(original_data)
    mask_values[sw_mask] = input_data[sw_mask]
    mask_values[non_sw_mask] = input_data[non_sw_mask]
    
    # 计算显示范围
    mask_vmax = np.percentile(np.abs(mask_values[mask_values != 0]), percentile) if np.any(mask_values != 0) else 1.0
    
    print(f"掩码值范围: [{-mask_vmax:.6f}, {mask_vmax:.6f}]")
    
    # 创建图像
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.imshow(
        mask_values, 
        aspect='auto', 
        cmap='seismic', 
        vmin=-mask_vmax, 
        vmax=mask_vmax, 
        interpolation='bilinear'
    )
    
    # 去除所有装饰
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 保存图像
    if save_prefix:
        output_path = f'{save_prefix}.png'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"已保存掩码值图像: {output_path}")
    
    return fig


# # ========== 使用示例 ==========
if __name__ == "__main__":
    
#     # ========== 示例1: 绘制单个npy文件 ==========
#     print("\n" + "="*70)
#     print("示例1: 绘制单个地震数据")
#     print("="*70)
    
#     single_npy_path = r"D:\桌面\面波压制\方法对比\U-Net3\denoised_shot_50.npy"
#     single_output = r"D:\桌面\面波压制\可视化结果\单个数据"
    
#     fig1 = visualize_npy_seismic_clean(
#         npy_path=single_npy_path,
#         save_prefix=single_output,
#         dpi=600,
#         figsize=(10, 18),
#         cmap='seismic',
#         use_percentile=True,
#         percentile=99
#     )
    
    # ========== 示例2: 批量绘制多个npy文件（使用统一vmax，与原代码逻辑一致）==========
    print("\n" + "="*70)
    print("示例2: 批量绘制多个地震数据（统一显示范围）")
    print("="*70)
    
    data_dir = r"D:\桌面\面波压制\Synthetic data test2\DMSSL"
    output_dir = r"D:\桌面\面波压制\图片汇总\流程图"
    
    npy_files = [
        os.path.join(data_dir, "original_shot_0.npy"),
        os.path.join(data_dir, "input_shot_0.npy"),
        os.path.join(data_dir, "denoised_shot_0.npy"),
        os.path.join(data_dir, "noise_0.npy")
    ]
    
    titles = ["原始数据", "网络输入", "去噪结果", "噪声"]
    
    fig_list = visualize_multiple_npy_clean(
        npy_paths=npy_files,
        save_prefix=os.path.join(output_dir, "batch"),
        dpi=600,
        figsize=(10, 18),
        cmaps=['seismic', 'seismic', 'seismic', 'seismic'],
        use_percentile=True,
        percentile=99,
        titles=titles,
        use_unified_vmax=True,  # 使用统一的vmax
        reference_idx=0  # 基于第0个文件（原始数据）计算vmax
    )
    
#     # ========== 示例3: 绘制掩码数据 ==========
#     print("\n" + "="*70)
#     print("示例3: 绘制掩码数据")
#     print("="*70)
    
#     mask_path = os.path.join(data_dir, "sw_mask_shot_50.npy")
#     mask_output = os.path.join(output_dir, "mask")
    
#     fig_mask = visualize_mask_npy_clean(
#         mask_path=mask_path,
#         save_prefix=mask_output,
#         dpi=600,
#         figsize=(10, 18),
#         cmap='bwr'
#     )
    
    # ========== 示例4: 绘制RGB组合掩码（如果有两个掩码文件）==========
    # print("\n" + "="*70)
    # print("示例4: 绘制RGB组合掩码")
    # print("="*70)
    
    # sw_mask_path = os.path.join(data_dir, "sw_mask_shot_50.npy")
    # non_sw_mask_path = os.path.join(data_dir, "non_sw_mask_shot_50.npy")
    # rgb_output = os.path.join(output_dir, "rgb_mask")
    
    # fig_rgb = visualize_rgb_mask_clean(
    #     sw_mask_path=sw_mask_path,
    #     non_sw_mask_path=non_sw_mask_path,
    #     save_prefix=rgb_output,
    #     dpi=600,
    #     figsize=(10, 18)
    # )
    
    # ========== 示例5: 绘制掩码值图 ==========
    # print("\n" + "="*70)
    # print("示例5: 绘制掩码值图")
    # print("="*70)
    
    # original_path = os.path.join(data_dir, "original_shot_50.npy")
    # input_path = os.path.join(data_dir, "input_shot_50.npy")
    # mask_values_output = os.path.join(output_dir, "mask_values")
    
    # fig_mask_values = visualize_mask_values_clean(
    #     original_data_path=original_path,
    #     input_data_path=input_path,
    #     sw_mask_path=sw_mask_path,
    #     non_sw_mask_path=non_sw_mask_path,
    #     save_prefix=mask_values_output,
    #     dpi=600,
    #     figsize=(10, 18),
    #     percentile=99
    # )
    
    # 显示所有图像
    plt.show()
    
    print("\n" + "="*70)
    print("所有图像绘制完成！")
    print("="*70)