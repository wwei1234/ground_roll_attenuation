import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from sklearn.utils.extmath import randomized_svd
import pywt
from tqdm import tqdm

class SeismicInterpolation:
    """地震数据传统插值方法实现"""
    
    def __init__(self, data, mask, use_smart_init=True):
        """
        初始化
        
        Parameters:
        -----------
        data : np.ndarray
            含面波的炮集数据, shape: (n_samples, n_traces)
        mask : np.ndarray
            面波区域掩码, shape: (n_samples, n_traces)
            1表示面波区域(需要插值), 0表示有效区域
        use_smart_init : bool
            是否使用智能初始化(邻域插值),False则直接填0
        """
        self.data = data.astype(np.float32)
        self.mask = mask.astype(np.float32)
        
        # 构建待插值数据
        if use_smart_init:
            print("使用邻域线性插值初始化面波区域...")
            self.data_masked = self._initialize_masked_region(self.data, self.mask)
        else:
            print("使用零值初始化面波区域...")
            self.data_masked = self.data * (1 - self.mask)
        
        print(f"数据形状: {self.data.shape}")
        print(f"面波区域占比: {self.mask.sum() / self.mask.size * 100:.2f}%")
    
    
    def _initialize_masked_region(self, data, mask):
        """
        用邻域信息初始化面波区域,避免硬0边界
        
        策略: 沿时间方向对每道进行线性插值
        
        Parameters:
        -----------
        data : np.ndarray
            原始数据
        mask : np.ndarray
            掩码 (1=面波区域, 0=有效区域)
            
        Returns:
        --------
        result : np.ndarray
            初始化后的数据
        """
        result = data.copy()
        
        # 沿时间方向(每道)进行线性插值
        for trace_idx in range(data.shape[1]):
            trace = data[:, trace_idx]
            trace_mask = mask[:, trace_idx]
            
            masked_samples = np.where(trace_mask == 1)[0]
            valid_samples = np.where(trace_mask == 0)[0]
            
            if len(masked_samples) > 0 and len(valid_samples) > 1:
                # 使用有效样本点进行线性插值
                result[masked_samples, trace_idx] = np.interp(
                    masked_samples,
                    valid_samples,
                    trace[valid_samples]
                )
            elif len(masked_samples) > 0:
                # 如果整道都是面波,填充为0
                result[masked_samples, trace_idx] = 0
        
        return result
    
    
    def curvelet_interpolation(self, n_iterations=20, threshold_decay=0.95, verbose=True):
        """
        基于Curvelet变换的迭代插值
        
        由于真正的Curvelet变换需要CurveLab库(较难安装),
        这里使用小波变换的多方向分解作为近似替代
        
        Parameters:
        -----------
        n_iterations : int
            迭代次数
        threshold_decay : float
            阈值衰减率 (推荐: 0.90-0.97)
        verbose : bool
            是否打印详细信息
            
        Returns:
        --------
        result : np.ndarray
            插值后的数据
        """
        if verbose:
            print("\n" + "="*50)
            print("Curvelet域插值 (使用小波多方向分解近似)")
            print("="*50)
            print(f"参数: n_iterations={n_iterations}, threshold_decay={threshold_decay}")
        
        # 初始化
        result = self.data_masked.copy()
        residuals = []
        
        for iteration in tqdm(range(n_iterations), desc="迭代进度", disable=not verbose):
            # 1. 前向小波变换 (使用2D小波作为Curvelet的近似)
            coeffs = pywt.wavedec2(result, 'db4', level=3)
            
            # 2. 软阈值处理 (稀疏性约束)
            threshold = np.std(result) * threshold_decay ** iteration
            coeffs_thresh = self._soft_threshold_coeffs(coeffs, threshold)
            
            # 3. 反变换
            result_new = pywt.waverec2(coeffs_thresh, 'db4')
            
            # 确保形状一致
            if result_new.shape != result.shape:
                result_new = result_new[:result.shape[0], :result.shape[1]]
            
            # 4. 数据一致性约束: 面波区域用插值结果,有效区域保持原始数据
            result = result_new * self.mask + self.data * (1 - self.mask)
            
            # 5. 计算残差 (仅在面波区域)
            residual = np.linalg.norm((result - result_new) * self.mask)
            residuals.append(residual)
            
            if verbose and iteration % 5 == 0:
                print(f"  迭代 {iteration+1}: 面波区域残差 = {residual:.6f}, 阈值 = {threshold:.6f}")
        
        if verbose:
            print(f"✓ Curvelet插值完成")
            print(f"  最终残差: {residuals[-1]:.6f}")
            if len(residuals) > 5:
                conv_rate = (residuals[-1] - residuals[-6]) / residuals[-6]
                print(f"  最后5次迭代收敛率: {conv_rate*100:.2f}%")
        
        return result
    
    
    def _soft_threshold_coeffs(self, coeffs, threshold):
        """对小波系数进行软阈值处理"""
        coeffs_thresh = [coeffs[0]]  # 近似系数不处理
        
        for detail_level in coeffs[1:]:
            thresh_level = []
            for detail in detail_level:
                thresh_detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                thresh_level.append(thresh_detail)
            coeffs_thresh.append(tuple(thresh_level))
        
        return coeffs_thresh
    
    
    def low_rank_matrix_completion(self, rank=None, n_iterations=50, verbose=True):
        """
        基于低秩矩阵补全的插值 (使用交替最小化)
        
        假设地震数据矩阵具有低秩结构
        
        Parameters:
        -----------
        rank : int or None
            目标秩, 如果为None则自动估计
        n_iterations : int
            迭代次数
        verbose : bool
            是否打印详细信息
            
        Returns:
        --------
        result : np.ndarray
            插值后的数据
        """
        if verbose:
            print("\n" + "="*50)
            print("低秩矩阵补全插值")
            print("="*50)
        
        # 自动估计秩
        if rank is None:
            # 使用观测数据的SVD估计有效秩
            U, s, Vt = randomized_svd(self.data_masked, n_components=min(50, min(self.data.shape)-1))
            # 选择保留90%能量的秩
            cumsum_energy = np.cumsum(s**2) / np.sum(s**2)
            rank = np.searchsorted(cumsum_energy, 0.90) + 1
            if verbose:
                print(f"自动估计秩: {rank} (保留90%能量)")
        else:
            if verbose:
                print(f"使用指定秩: {rank}")
        
        # 初始化: 使用截断SVD
        U, s, Vt = randomized_svd(self.data_masked, n_components=rank)
        result = U @ np.diag(s) @ Vt
        
        changes = []
        
        # 迭代优化
        for iteration in tqdm(range(n_iterations), desc="迭代进度", disable=not verbose):
            # 1. 数据一致性约束: 面波区域用插值结果,有效区域保持原始数据
            result = result * self.mask + self.data * (1 - self.mask)
            
            # 2. 低秩投影
            U, s, Vt = randomized_svd(result, n_components=rank)
            result_new = U @ np.diag(s) @ Vt
            
            # 3. 计算变化 (仅在面波区域)
            mask_change = np.linalg.norm((result - result_new) * self.mask)
            total_norm = np.linalg.norm(result * self.mask)
            relative_change = mask_change / (total_norm + 1e-10)
            changes.append(relative_change)
            
            if verbose and iteration % 10 == 0:
                print(f"  迭代 {iteration+1}: 面波区域相对变化 = {relative_change:.6f}")
            
            result = result_new
        
        # 最后一次数据约束
        result = result * self.mask + self.data * (1 - self.mask)
        
        if verbose:
            print(f"✓ 低秩矩阵补全完成")
            print(f"  最终相对变化: {changes[-1]:.6f}")
            if len(changes) > 5:
                conv_rate = (changes[-1] - changes[-6]) / (changes[-6] + 1e-10)
                print(f"  最后5次迭代收敛率: {conv_rate*100:.2f}%")
        
        return result
    
    
    def pocs_interpolation(self, n_iterations=30, init_percentile=95, decay_rate=1.5, verbose=True):
        """
        POCS (Projection Onto Convex Sets) 迭代插值
        在F-K域进行稀疏约束
        
        Parameters:
        -----------
        n_iterations : int
            迭代次数
        init_percentile : float
            初始阈值百分位 (推荐: 90-98)
        decay_rate : float
            阈值衰减速度 (推荐: 1.0-2.0)
        verbose : bool
            是否打印详细信息
            
        Returns:
        --------
        result : np.ndarray
            插值后的数据
        """
        if verbose:
            print("\n" + "="*50)
            print("POCS (F-K域稀疏约束) 插值")
            print("="*50)
            print(f"参数: n_iterations={n_iterations}, init_percentile={init_percentile}, decay_rate={decay_rate}")
        
        result = self.data_masked.copy()
        residuals = []
        
        for iteration in tqdm(range(n_iterations), desc="迭代进度", disable=not verbose):
            # 1. 前向FFT到F-K域
            fk_spectrum = fft2(result)
            
            # 2. 稀疏性约束: 软阈值 (阈值逐渐降低)
            current_percentile = max(50, init_percentile - iteration * decay_rate)
            threshold = np.percentile(np.abs(fk_spectrum), current_percentile)
            fk_spectrum = np.sign(fk_spectrum) * np.maximum(np.abs(fk_spectrum) - threshold, 0)
            
            # 3. 反变换到时空域
            result_new = np.real(ifft2(fk_spectrum))
            
            # 4. 数据一致性约束: 面波区域用插值结果,有效区域保持原始数据
            result = result_new * self.mask + self.data * (1 - self.mask)
            
            # 5. 计算残差 (仅在面波区域)
            residual = np.linalg.norm((result - result_new) * self.mask)
            residuals.append(residual)
            
            if verbose and iteration % 10 == 0:
                print(f"  迭代 {iteration+1}: 面波区域残差 = {residual:.6f}, 百分位 = {current_percentile:.1f}")
        
        if verbose:
            print(f"✓ POCS插值完成")
            print(f"  最终残差: {residuals[-1]:.6f}")
            if len(residuals) > 5:
                conv_rate = (residuals[-1] - residuals[-6]) / residuals[-6]
                print(f"  最后5次迭代收敛率: {conv_rate*100:.2f}%")
        
        return result
    
    
    def evaluate_results(self, results_dict, ground_truth=None):
        """
        评估插值结果
        
        Parameters:
        -----------
        results_dict : dict
            {方法名: 插值结果}
        ground_truth : np.ndarray or None
            真实无面波数据 (如果有的话)
        """
        print("\n" + "="*50)
        print("插值结果评估")
        print("="*50)
        
        for method_name, result in results_dict.items():
            print(f"\n{method_name}:")
            
            # 在面波区域的重建质量
            reconstructed_region = result * self.mask
            original_in_mask = self.data * self.mask
            
            # 能量恢复比
            energy_original = np.sum(original_in_mask ** 2)
            energy_reconstructed = np.sum(reconstructed_region ** 2)
            energy_ratio = energy_reconstructed / (energy_original + 1e-10)
            print(f"  面波区域能量恢复比: {energy_ratio:.4f}")
            
            # 如果有ground truth
            if ground_truth is not None:
                # 仅在面波区域计算误差
                mse = np.mean((result * self.mask - ground_truth * self.mask) ** 2)
                psnr = 10 * np.log10(np.max(ground_truth)**2 / (mse + 1e-10))
                print(f"  面波区域MSE: {mse:.6f}")
                print(f"  面波区域PSNR: {psnr:.2f} dB")
                
                # 整体相关系数
                corr = np.corrcoef(result.flatten(), ground_truth.flatten())[0, 1]
                print(f"  整体相关系数: {corr:.4f}")
    
    
    def plot_convergence(self, results_dict):
        """
        绘制收敛曲线(需要在插值方法中记录residuals)
        
        Parameters:
        -----------
        results_dict : dict
            {方法名: (result, residuals)}
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 残差曲线
        for method_name, (result, residuals) in results_dict.items():
            if residuals is not None and len(residuals) > 0:
                ax1.plot(residuals, label=method_name, linewidth=2)
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Residual (in masked region)', fontsize=12)
        ax1.set_title('Convergence Curve', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 右图: 收敛率
        for method_name, (result, residuals) in results_dict.items():
            if residuals is not None and len(residuals) > 5:
                conv_rates = []
                for i in range(5, len(residuals)):
                    rate = abs(residuals[i] - residuals[i-5]) / (residuals[i-5] + 1e-10)
                    conv_rates.append(rate)
                ax2.plot(range(5, len(residuals)), conv_rates, label=method_name, linewidth=2)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Convergence Rate (5-iter window)', fontsize=12)
        ax2.set_title('Convergence Rate', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
        
        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ 收敛曲线已保存至: convergence_analysis.png")
        plt.show()
    
    
    def visualize_results(self, results_dict, save_path=None):
        """
        可视化对比结果
        
        Parameters:
        -----------
        results_dict : dict
            {方法名: 插值结果} 或 {方法名: (插值结果, residuals)}
        save_path : str or None
            保存路径
        """
        # 提取结果(如果是tuple则取第一个元素)
        results_only = {}
        for k, v in results_dict.items():
            results_only[k] = v[0] if isinstance(v, tuple) else v
        
        n_methods = len(results_only)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods+1), 10))
        
        # 显示原始含面波数据
        vmin, vmax = np.percentile(self.data, [2, 98])
        
        axes[0, 0].imshow(self.data, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Original (with surface wave)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Time samples', fontsize=10)
        
        axes[1, 0].imshow(self.mask, cmap='gray', aspect='auto')
        axes[1, 0].set_title('Surface wave mask', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Time samples', fontsize=10)
        axes[1, 0].set_xlabel('Trace number', fontsize=10)
        
        # 显示各方法结果
        for idx, (method_name, result) in enumerate(results_only.items()):
            axes[0, idx+1].imshow(result, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, idx+1].set_title(f'{method_name}', fontsize=12, fontweight='bold')
            
            # 显示差异图 (原始 - 插值结果)
            diff = self.data - result
            vmin_diff, vmax_diff = np.percentile(diff, [2, 98])
            axes[1, idx+1].imshow(diff, cmap='seismic', aspect='auto', 
                                  vmin=vmin_diff, vmax=vmax_diff)
            axes[1, idx+1].set_title(f'Removed (Original - {method_name})', fontsize=12)
            axes[1, idx+1].set_xlabel('Trace number', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ 结果图已保存至: {save_path}")
        
        plt.show()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 1. 加载数据
    print("加载数据...")
    data = np.load('D:\桌面\面波压制\Synthetic data test 2\DMSSL\original_shot_0.npy')[100:500, 5:43]  # 含面波炮集
    mask = np.load('D:\桌面\面波压制\Synthetic data test 2\DMSSL\sw_mask_shot_0.npy')[100:500, 5:43]              # 面波掩码 (1=面波区域)
    
    # 可选: 如果有无面波的真实数据用于评估
    # ground_truth = np.load('shot_gather_without_surface_wave.npy')
    ground_truth = None
    
    # 2. 初始化插值器
    # use_smart_init=True: 使用邻域插值初始化 (推荐)
    # use_smart_init=False: 直接填0初始化
    interpolator = SeismicInterpolation(data, mask, use_smart_init=True)
    
    # 3. 运行各种插值方法
    results = {}
    
    # 方法1: Curvelet域插值 (小波近似)
    # 参数调优建议:
    # - 高信噪比数据: n_iterations=20, threshold_decay=0.92
    # - 低信噪比数据: n_iterations=25, threshold_decay=0.96
    # - 复杂构造: n_iterations=30, threshold_decay=0.95
    results['Curvelet-like'] = interpolator.curvelet_interpolation(
        n_iterations=20, 
        threshold_decay=0.95,
        verbose=True
    )
    
    # 方法2: 低秩矩阵补全
    # 参数调优建议:
    # - rank=None: 自动估计 (推荐先尝试)
    # - rank=15: 适用于简单构造
    # - rank=25: 适用于复杂构造
    # - n_iterations=50: 标准迭代次数
    results['Low-Rank'] = interpolator.low_rank_matrix_completion(
        rank=None,  # 自动估计
        n_iterations=50,
        verbose=True
    )
    
    # 方法3: POCS (F-K域)
    # 参数调优建议:
    # - init_percentile=95: 初始保留5%强系数 (推荐)
    # - decay_rate=1.5: 阈值衰减速度 (推荐)
    # - n_iterations=30: 标准迭代次数
    results['POCS (F-K)'] = interpolator.pocs_interpolation(
        n_iterations=30,
        init_percentile=95,
        decay_rate=1.5,
        verbose=True
    )
    
    # 4. 评估结果
    interpolator.evaluate_results(results, ground_truth)
    
    # 5. 可视化对比
    interpolator.visualize_results(results, save_path='interpolation_comparison.png')
    
    # 6. 保存结果
    print("\n保存插值结果...")
    for method_name, result in results.items():
        filename = f"result_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}.npy"
        np.save(filename, result)
        print(f"  ✓ {filename}")
    
    print("\n" + "="*50)
    print("所有处理完成!")
    print("="*50)
    
    # 7. (可选) 参数调优实验
    print("\n" + "="*50)
    print("运行参数调优实验...")
    print("="*50)
    
    # 测试不同threshold_decay对Curvelet方法的影响
    print("\n测试Curvelet方法的threshold_decay参数:")
    decay_values = [0.90, 0.93, 0.95, 0.97]
    decay_results = {}
    for decay in decay_values:
        print(f"\n  测试 threshold_decay = {decay}")
        result = interpolator.curvelet_interpolation(
            n_iterations=20, 
            threshold_decay=decay,
            verbose=False
        )
        decay_results[f'decay_{decay}'] = result
        
        # 简单评估
        energy_ratio = np.sum((result * mask) ** 2) / (np.sum((data * mask) ** 2) + 1e-10)
        print(f"    能量恢复比: {energy_ratio:.4f}")
    
    # 测试不同rank对低秩方法的影响
    print("\n测试低秩方法的rank参数:")
    rank_values = [10, 15, 20, 25]
    rank_results = {}
    for r in rank_values:
        print(f"\n  测试 rank = {r}")
        result = interpolator.low_rank_matrix_completion(
            rank=r, 
            n_iterations=30,
            verbose=False
        )
        rank_results[f'rank_{r}'] = result
        
        # 简单评估
        energy_ratio = np.sum((result * mask) ** 2) / (np.sum((data * mask) ** 2) + 1e-10)
        print(f"    能量恢复比: {energy_ratio:.4f}")
    
    print("\n✓ 参数调优实验完成")