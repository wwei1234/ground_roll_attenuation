import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
from matplotlib.path import Path  # 新增：用于高效掩码生成
import segyio
import os
from U_Net_CBAM import UNet  # 保持你的模型导入

class ManualSurfaceWaveAnnotator:
    """手动标注面波区域的交互式工具（优化版：高效掩码 + 流畅交互）"""
    
    def __init__(self, shot_gather, shot_center=22, dt=0.00025):
        self.shot_gather = shot_gather
        self.shot_center = shot_center
        self.dt = dt
        self.time_samples, self.trace_num = shot_gather.shape
        
        self.polygons = []              # 已完成的多边形
        self.current_polygon = []       # 当前正在绘制的多边形
        self.polygon_patches = []       # 已完成的多边形patch
        self.current_line = None        # 当前连线（用于动态更新）
        self.current_points = []        # 当前顶点（用于删除）
        
        self.sw_mask = None
        self.finished = False
        
    def create_mask_from_polygons(self):
        """高效生成掩码：使用 matplotlib.path 批量判断点是否在多边形内"""
        mask = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        
        if not self.polygons:
            return mask
        
        # 生成所有网格点 (x=道号, y=时间采样点)
        xx, yy = np.meshgrid(np.arange(self.trace_num), np.arange(self.time_samples))
        points = np.stack((xx.ravel(), yy.ravel()), axis=1)
        
        for polygon in self.polygons:
            if len(polygon) < 3:
                continue
            path = Path(polygon, closed=True)
            inside = path.contains_points(points, radius=1e-9)
            mask |= inside.reshape(self.time_samples, self.trace_num)
        
        return mask
    
    def onclick(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        if event.button == 1:  # 左键添加顶点
            x, y = event.xdata, event.ydata
            self.current_polygon.append((x, y))
            
            # 绘制新点
            pt = self.ax.plot(x, y, 'ro', markersize=8)[0]
            self.current_points.append(pt)
            
            # 更新连线
            if self.current_line:
                self.current_line.remove()
            if len(self.current_polygon) > 1:
                poly_array = np.array(self.current_polygon)
                self.current_line, = self.ax.plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
            
            self.fig.canvas.draw_idle()
            
        elif event.button == 3:  # 右键完成当前多边形
            if len(self.current_polygon) >= 3:
                self.polygons.append(self.current_polygon.copy())
                
                # 添加永久多边形
                patch = Polygon(np.array(self.current_polygon), fill=True, alpha=0.3,
                                facecolor='red', edgecolor='red', linewidth=2)
                self.ax.add_patch(patch)
                self.polygon_patches.append(patch)
                
                print(f"✓ 完成多边形 {len(self.polygons)}，顶点数: {len(self.current_polygon)}")
                
                # 清除当前临时绘制
                self.clear_current_temp()
                self.current_polygon = []
                
                self.fig.canvas.draw_idle()
            else:
                print("⚠ 多边形至少需要3个顶点")
    
    def clear_current_temp(self):
        """清除当前正在绘制的临时线和点"""
        if self.current_line:
            self.current_line.remove()
            self.current_line = None
        for pt in self.current_points:
            pt.remove()
        self.current_points = []
    
    def onkey(self, event):
        if event.key == 'z' and self.current_polygon:  # 撤销最后一个点
            self.current_polygon.pop()
            self.clear_current_temp()
            # 重新绘制当前多边形
            if self.current_polygon:
                poly_array = np.array(self.current_polygon)
                self.current_line, = self.ax.plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
                for x, y in self.current_polygon:
                    pt = self.ax.plot(x, y, 'ro', markersize=8)[0]
                    self.current_points.append(pt)
            self.fig.canvas.draw_idle()
            print("↶ 撤销最后一个点")
            
        elif event.key == 'c':  # 清除当前多边形
            self.current_polygon = []
            self.clear_current_temp()
            self.fig.canvas.draw_idle()
            print("✗ 清除当前多边形")
            
        elif event.key == 'delete' and self.polygons:  # 删除最后一个完成的多边形
            self.polygons.pop()
            patch = self.polygon_patches.pop()
            patch.remove()
            self.fig.canvas.draw_idle()
            print(f"✗ 删除多边形，剩余 {len(self.polygons)} 个")
    
    def on_finish(self, event):
        if self.polygons:
            print(f"\n✓ 标注完成！共标注 {len(self.polygons)} 个面波区域")
            self.sw_mask = self.create_mask_from_polygons()
            self.finished = True
            plt.close(self.fig)
        else:
            print("⚠ 请至少标注一个面波区域")
    
    def on_skip(self, event):
        print("\n⊘ 跳过手动标注（全图无面波）")
        self.sw_mask = np.zeros((self.time_samples, self.trace_num), dtype=bool)
        self.finished = True
        plt.close(self.fig)
    
    def annotate(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.15)
        
        vmax = np.percentile(np.abs(self.shot_gather), 99)
        self.ax.imshow(self.shot_gather, aspect='auto', cmap='seismic',
                       vmin=-vmax, vmax=vmax, interpolation='bilinear')
        self.ax.set_title('手动标注面波区域\n左键添加顶点 | 右键完成多边形 | Z撤销 | C清除当前 | Delete删上一个', 
                          fontsize=12, fontweight='bold')
        self.ax.set_xlabel('道号')
        self.ax.set_ylabel('时间采样点')
        self.ax.axvline(self.shot_center, color='lime', linestyle='--', linewidth=2, label='炮点')
        self.ax.legend(loc='upper right')
        
        # 按钮
        ax_finish = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_skip = plt.axes([0.81, 0.05, 0.1, 0.05])
        btn_finish = Button(ax_finish, '完成标注')
        btn_skip = Button(ax_skip, '跳过')
        btn_finish.on_clicked(self.on_finish)
        btn_skip.on_clicked(self.on_skip)
        
        # 事件连接
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        
        print("\n" + "="*60)
        print("手动标注面波区域 - 操作说明:")
        print("="*60)
        print("🖱  左键点击: 添加顶点")
        print("🖱  右键点击: 完成当前多边形")
        print("⌨  Z键: 撤销最后一个顶点")
        print("⌨  C键: 清除当前多边形")
        print("⌨  Delete键: 删除上一个完成的多边形")
        print("🔘 完成标注/跳过 按钮: 结束标注")
        print("="*60 + "\n")
        
        plt.show()
        
        return self.sw_mask if self.finished else None


class SeismicDenoiser:
    """简化版地震数据去噪器（仅支持手动标注面波）"""
    
    def __init__(self, model_path, device=None, fixed_mask_value=0.5):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = UNet(in_channels=1, num_classes=1, base_c=64)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.fixed_mask_value = fixed_mask_value  # 面波区替换为高斯噪声的标准差倍数
    
    def read_segy(self, data_dir, shotnum=0):
        with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)
            if shotnum:
                shot_num = shotnum 
            else:
                shot_num = len(set(sourceX))
            len_shot = trace_num // shot_num
            time = f.trace[0].shape[0]
            data = np.zeros((shot_num, time, len_shot))
            for j in range(shot_num):
                data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
            return data
    
    def normalize_traces_per_trace(self, shot_gather):
        """逐道归一化"""
        normalized = np.copy(shot_gather)
        for i in range(normalized.shape[1]):
            trace = normalized[:, i]
            max_amp = np.max(np.abs(trace))
            if max_amp > 0:
                normalized[:, i] = trace / max_amp
        return normalized
    
    def replace_surface_wave_with_mask(self, shot_gather, sw_mask):
        """将面波区域替换为高斯噪声"""
        gaussian_noise = np.random.normal(0, self.fixed_mask_value / 3.0, 
                                          size=shot_gather.shape)
        gaussian_noise = np.clip(gaussian_noise, -self.fixed_mask_value, self.fixed_mask_value)
        
        masked_shot = shot_gather.copy()
        masked_shot[sw_mask] = gaussian_noise[sw_mask]
        return masked_shot
    
    def boost_surface_wave_energy(self, denoised_data, sw_mask, energy_boost_factor=1.5):
        boosted = denoised_data.copy()
        boosted[sw_mask] *= energy_boost_factor
        return boosted
    
    def preprocess_shot(self, shot_gather, norm=True):
        if norm:
            return self.normalize_traces_per_trace(shot_gather)
        else:
            return shot_gather.copy()
    
    @torch.no_grad()
    def denoise_shot(self, shot_gather, sw_mask=None,
                     replace_non_sw_with_input=True,
                     boost_sw_energy=True,
                     energy_boost_factor=1.5):
        input_tensor = torch.from_numpy(shot_gather).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        output = self.model(input_tensor)
        denoised = output.squeeze().cpu().numpy()
        
        if boost_sw_energy and sw_mask is not None:
            denoised = self.boost_surface_wave_energy(denoised, sw_mask, energy_boost_factor)
        
        if replace_non_sw_with_input and sw_mask is not None:
            denoised[~sw_mask] = shot_gather[~sw_mask]
        
        return denoised
    
    def save_shot_to_npy(self, data, output_path):
        np.save(output_path, data)
    
    def plot_comparison(self, original, network_input, denoised, sw_mask, 
                        shot_idx, shot_center=22, manual_annotation=True,
                        save_path=None, figsize=(18, 12)):
        # （绘图部分保持不变，仅改了个标题文字）
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        time_samples, trace_num = original.shape
        vmax = np.percentile(np.abs(original), 99)
        
        im0 = axes[0, 0].imshow(original, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0, 0].set_title(f'预处理后原始数据 - 第{shot_idx}炮')
        axes[0, 0].set_xlabel('道号'); axes[0, 0].set_ylabel('时间采样点')
        axes[0, 0].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(sw_mask, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        title = '面波区域标识（手动标注）' if manual_annotation else '面波区域标识'
        axes[0, 1].set_title(title)
        axes[0, 1].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[0, 2].imshow(network_input, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[0, 2].set_title('网络输入（面波区已替换）')
        axes[0, 2].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im2, ax=axes[0, 2])
        
        axes[1, 0].imshow(network_input, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax)
        axes[1, 0].imshow(np.ma.masked_where(~sw_mask, sw_mask), aspect='auto', cmap='Reds', alpha=0.4)
        axes[1, 0].set_title('网络输入 + 面波区域叠加')
        axes[1, 0].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        
        im5 = axes[1, 1].imshow(denoised, aspect='auto', cmap='seismic',
                                vmin=-vmax, vmax=vmax, interpolation='bilinear')
        axes[1, 1].set_title('最终去噪结果', fontweight='bold', color='green')
        axes[1, 1].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im5, ax=axes[1, 1])
        
        total_diff = original - denoised
        vmax_diff = np.percentile(np.abs(total_diff), 99)
        im7 = axes[1, 2].imshow(total_diff, aspect='auto', cmap='seismic',
                                vmin=-vmax_diff, vmax=vmax_diff)
        axes[1, 2].set_title('总体差异（原始 - 最终）')
        axes[1, 2].axvline(shot_center, color='lime', linestyle='--', linewidth=2)
        plt.colorbar(im7, ax=axes[1, 2])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # ========== 配置 ==========
    model_path = r"D:\桌面\面波压制\model\model_field&syn_1111\model_epoch_500.pth"
    input_segy_path = r"D:\桌面\深层数据面波压制\sgy\real_data_gather182.sgy"
    output_dir = r"D:\桌面\深层数据面波压制\result\DMSSL_manual"
    os.makedirs(output_dir, exist_ok=True)
    
    USE_MANUAL_ANNOTATION = True          # 你现在固定为 True
    USE_MASK_FOR_SURFACE_WAVE = True      # 面波区替换为噪声
    BOOST_SURFACE_WAVE_ENERGY = True      # 是否轻微增强面波残留
    ENERGY_BOOST_FACTOR = 1.3             # 建议 1.2~1.8
    
    target_shot_idx = 0
    shot_center = 60
    shotnum = 1
    max_time_samples = 1200
    fixed_mask_value = 0.5                # 噪声幅度
    
    # ========== 初始化（参数大幅简化）==========
    denoiser = SeismicDenoiser(
        model_path=model_path,
        fixed_mask_value=fixed_mask_value
    )
    
    # ========== 读取数据 ==========
    seismic_data = denoiser.read_segy(input_segy_path, shotnum=shotnum)
    seismic_data = seismic_data[:, :max_time_samples, :]
    single_shot = seismic_data[target_shot_idx]
    preprocessed_data = denoiser.preprocess_shot(single_shot, norm=True)
    
    # ========== 手动标注面波 ==========
    print("\n启动手动标注模式...")
    annotator = ManualSurfaceWaveAnnotator(preprocessed_data, shot_center=shot_center)
    sw_mask = annotator.annotate()
    
    if sw_mask is None:
        print("标注被取消，程序退出")
        exit()
    
    # ========== 准备网络输入 ==========
    if USE_MASK_FOR_SURFACE_WAVE:
        network_input = denoiser.replace_surface_wave_with_mask(preprocessed_data, sw_mask)
    else:
        network_input = preprocessed_data.copy()
    
    # ========== 去噪 ==========
    denoised_shot = denoiser.denoise_shot(
        network_input,
        sw_mask=sw_mask,
        replace_non_sw_with_input=True,
        boost_sw_energy=BOOST_SURFACE_WAVE_ENERGY,
        energy_boost_factor=ENERGY_BOOST_FACTOR
    )
    
    # ========== 保存 ==========
    base_name = f"shot_{target_shot_idx}"
    denoiser.save_shot_to_npy(denoised_shot, os.path.join(output_dir, f"denoised_{base_name}.npy"))
    denoiser.save_shot_to_npy(preprocessed_data, os.path.join(output_dir, f"original_{base_name}.npy"))
    denoiser.save_shot_to_npy(network_input, os.path.join(output_dir, f"input_{base_name}.npy"))
    denoiser.save_shot_to_npy(sw_mask, os.path.join(output_dir, f"mask_{base_name}.npy"))
    
    # ========== 绘图 ==========
    comparison_path = os.path.join(output_dir, f"comparison_{base_name}.png")
    denoiser.plot_comparison(
        preprocessed_data, network_input, denoised_shot, sw_mask,
        target_shot_idx, shot_center, manual_annotation=True,
        save_path=comparison_path
    )
    
    print(f"\n✓ 处理完成！结果保存在: {output_dir}")