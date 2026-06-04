"""
基于手动标注掩码的面波压制推理脚本
=====================================
- 输入: npy格式炮集 + npy格式手动mask
- 网络: UNet（加载训练好的权重）
- 流程: 预处理 → 替换面波区域为高斯噪声 → 网络推理 → 后处理 → 保存 & 可视化
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from U_Net import UNet


# ============================================================
#  ★ 配置参数（修改这里）
# ============================================================
CONFIG = {
    # ---------- 路径 ----------
    "model_path"   : r"U-Net_3_7\best_model.pth",   # 训练好的模型权重
    "gather_path"  : r"面波压制结果\U-Net_139_2\real_data_gather139_denoised_denoised.npy",  # 单炮 npy 文件
    "mask_path"    : r"面波压制结果\shot139_3\real_data_gather139_denoised_denoised_sw_mask.npy",                # 对应 mask npy 文件
    "output_dir"   : r"面波压制结果\U-Net_139_3",                  # 结果保存目录

    # ---------- 预处理 ----------
    "normalize"         : True,    # 是否逐道归一化
    "max_time_samples"  : None,    # 截断时间采样点，None 表示不截断

    # ---------- 面波替换 ----------
    "fixed_mask_value"  : 0.5,     # 高斯噪声幅度上限（与训练时一致）

    # ---------- 后处理 ----------
    "replace_non_sw_with_input" : True,   # 非面波区域用原始输入替换网络输出
    "boost_sw_energy"           : False,  # 是否增强面波区域能量
    "energy_boost_factor"       : 1.0,    # 能量增强倍数

    # ---------- 设备 ----------
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
}


# ============================================================
#  工具函数
# ============================================================

def normalize_traces_per_trace(shot_gather: np.ndarray) -> np.ndarray:
    """逐道归一化（每道除以该道最大绝对值）"""
    normalized = shot_gather.copy()
    for i in range(normalized.shape[1]):
        trace = normalized[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized[:, i] = trace / max_amp
    return normalized


def replace_sw_with_gaussian(shot_gather: np.ndarray,
                              sw_mask: np.ndarray,
                              fixed_mask_value: float) -> np.ndarray:
    """将面波区域（sw_mask=1）替换为高斯噪声"""
    gaussian = np.random.normal(
        0, fixed_mask_value / 3.0,
        size=shot_gather.shape
    ).astype(np.float32)
    gaussian = np.clip(gaussian, -fixed_mask_value, fixed_mask_value)

    network_input = shot_gather.copy()
    network_input[sw_mask.astype(bool)] = gaussian[sw_mask.astype(bool)]
    return network_input


# ============================================================
#  推理器
# ============================================================

class ManualMaskDenoiser:
    """基于手动标注 mask 的面波压制推理器"""

    def __init__(self, model_path: str, device: str = None, fixed_mask_value: float = 0.5):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fixed_mask_value = fixed_mask_value

        # 加载模型
        self.model = UNet().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        # 兼容直接保存 state_dict 和保存 checkpoint dict 两种格式
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"加载 checkpoint（epoch={ckpt.get('epoch', '?')}，"
                  f"val_loss={ckpt.get('val_loss', '?'):.6f}）")
        else:
            self.model.load_state_dict(ckpt)
            print("加载 state_dict")
        self.model.eval()
        print(f"模型已加载至 {self.device}")

    def preprocess(self, shot_gather: np.ndarray,
                   normalize: bool = True,
                   max_time_samples: int = None) -> np.ndarray:
        """预处理：截断 + 归一化"""
        data = shot_gather.copy().astype(np.float32)
        if max_time_samples is not None:
            data = data[:max_time_samples, :]
        if normalize:
            data = normalize_traces_per_trace(data)
        return data

    @torch.no_grad()
    def denoise(self, shot_gather: np.ndarray,
                sw_mask: np.ndarray,
                replace_non_sw_with_input: bool = True,
                boost_sw_energy: bool = False,
                energy_boost_factor: float = 1.0) -> dict:
        """
        完整推理流程。

        Returns
        -------
        dict with keys:
            preprocessed   : 预处理后的原始数据
            network_input  : 送入网络的数据（面波区域替换为高斯）
            network_output : 网络原始输出
            denoised       : 后处理后的最终结果
            sw_mask        : 使用的面波掩码
            difference     : preprocessed - denoised
        """
        sw_mask_bool = sw_mask.astype(bool)

        # 1. 构建网络输入：面波区域换成高斯噪声
        network_input = replace_sw_with_gaussian(
            shot_gather, sw_mask_bool, self.fixed_mask_value
        )

        # 2. 推理
        inp_tensor = torch.from_numpy(network_input).float()
        inp_tensor = inp_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,T,X)
        out_tensor = self.model(inp_tensor)
        network_output = out_tensor.squeeze().cpu().numpy()  # (T,X)

        # 3. 后处理
        denoised = network_output.copy()

        # 非面波区域用原始输入替换（保留原始反射信息）
        if replace_non_sw_with_input:
            non_sw = ~sw_mask_bool
            denoised[non_sw] = shot_gather[non_sw]

        # 面波区域能量增强
        if boost_sw_energy and sw_mask_bool.any():
            denoised[sw_mask_bool] *= energy_boost_factor

        return {
            'preprocessed'  : shot_gather,
            'network_input' : network_input,
            'network_output': network_output,
            'denoised'      : denoised,
            'sw_mask'       : sw_mask,
            'difference'    : shot_gather - denoised,
        }

    def plot_results(self, results: dict, save_path: str = None, figsize=(20, 10)):
        """绘制 6 张对比子图"""
        preprocessed   = results['preprocessed']
        network_input  = results['network_input']
        network_output = results['network_output']
        denoised       = results['denoised']
        sw_mask        = results['sw_mask']
        difference     = results['difference']

        vmax     = np.percentile(np.abs(preprocessed), 99) or 1.0
        vmax_diff = np.percentile(np.abs(difference), 99) or 1.0

        kw_seis = dict(aspect='auto', cmap='seismic',
                       vmin=-vmax, vmax=vmax, interpolation='bilinear')
        kw_mask = dict(aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=1, interpolation='nearest')

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        titles = [
            ('1. 预处理后原始数据', kw_seis, preprocessed),
            ('2. 面波掩码（手动标注）', kw_mask, sw_mask.astype(float)),
            ('3. 网络输入', kw_seis, network_input),
            ('4. 网络原始输出', kw_seis, network_output),
            ('5. 最终去噪结果', kw_seis, denoised),
            ('6. 差异（原始 - 结果）',
             dict(aspect='auto', cmap='seismic',
                  vmin=-vmax_diff, vmax=vmax_diff, interpolation='bilinear'),
             difference),
        ]

        for ax, (title, kw, data) in zip(axes, titles):
            im = ax.imshow(data, **kw)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('道号')
            ax.set_ylabel('时间采样点')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 统计信息
        sw_ratio   = sw_mask.sum() / sw_mask.size * 100
        signal_ratio = (
            np.mean(np.abs(denoised[~sw_mask.astype(bool)])) /
            (np.mean(np.abs(preprocessed[~sw_mask.astype(bool)])) + 1e-10)
        ) * 100

        fig.suptitle(
            f'面波压制结果  |  面波占比: {sw_ratio:.1f}%  |  '
            f'非面波区域能量保留: {signal_ratio:.1f}%',
            fontsize=12, y=1.01
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"对比图已保存: {save_path}")
        else:
            plt.show()

        return fig

    def save_results(self, results: dict, output_dir: str, prefix: str = "shot"):
        """保存所有中间结果为 npy 文件"""
        os.makedirs(output_dir, exist_ok=True)
        save_map = {
            'preprocessed'  : f"{prefix}_preprocessed.npy",
            'network_input' : f"{prefix}_network_input.npy",
            'network_output': f"{prefix}_network_output.npy",
            'denoised'      : f"{prefix}_denoised.npy",
            'sw_mask'       : f"{prefix}_sw_mask.npy",
            'difference'    : f"{prefix}_difference.npy",
        }
        for key, fname in save_map.items():
            path = os.path.join(output_dir, fname)
            np.save(path, results[key])
        print(f"所有结果已保存至: {output_dir}")


# ============================================================
#  主流程
# ============================================================

def main():
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

    cfg = CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # ---------- 加载数据 ----------
    print(f"\n加载炮集: {cfg['gather_path']}")
    gather_raw = np.load(cfg['gather_path']).astype(np.float32)
    print(f"  原始形状: {gather_raw.shape}")

    print(f"加载掩码: {cfg['mask_path']}")
    sw_mask = np.load(cfg['mask_path'], allow_pickle=True)
    if sw_mask.ndim == 0:
        sw_mask = sw_mask.item()
    sw_mask = sw_mask.astype(np.uint8)
    print(f"  掩码形状: {sw_mask.shape}  面波占比: {sw_mask.mean()*100:.1f}%")

    # ---------- 初始化推理器 ----------
    denoiser = ManualMaskDenoiser(
        model_path       = cfg['model_path'],
        device           = cfg['device'],
        fixed_mask_value = cfg['fixed_mask_value'],
    )

    # ---------- 预处理 ----------
    gather = denoiser.preprocess(
        gather_raw,
        normalize        = cfg['normalize'],
        max_time_samples = cfg['max_time_samples'],
    )

    # mask 与 gather 对齐（以防 max_time_samples 截断了时间轴）
    if gather.shape != sw_mask.shape:
        print(f"  [警告] gather 形状 {gather.shape} 与 mask 形状 {sw_mask.shape} 不一致，"
              f"自动裁剪 mask")
        sw_mask = sw_mask[:gather.shape[0], :gather.shape[1]]

    print(f"\n预处理后形状: {gather.shape}")

    # ---------- 推理 ----------
    print("\n开始推理...")
    results = denoiser.denoise(
        gather,
        sw_mask,
        replace_non_sw_with_input = cfg['replace_non_sw_with_input'],
        boost_sw_energy           = cfg['boost_sw_energy'],
        energy_boost_factor       = cfg['energy_boost_factor'],
    )
    print("推理完成")

    # ---------- 保存 ----------
    gather_name = os.path.splitext(os.path.basename(cfg['gather_path']))[0]
    denoiser.save_results(results, cfg['output_dir'], prefix=gather_name)

    # ---------- 可视化 ----------
    fig_path = os.path.join(cfg['output_dir'], f"{gather_name}_comparison.png")
    denoiser.plot_results(results, save_path=fig_path)

    print(f"\n✓ 全部完成，结果保存在: {cfg['output_dir']}")


if __name__ == "__main__":
    main()