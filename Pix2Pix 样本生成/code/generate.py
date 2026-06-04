# generate.py
"""
使用训练好的 Pix2Pix 生成器，批量从 mask 生成含面波的炮集。

生成结果直接保存为 .npy 文件，可直接送入原有的面波识别网络进行训练扩充。
同时保存对应的 mask（作为标签），确保配对关系正确。
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import zoom

from model import build_unet_generator

# ==================== 用户配置区 ====================
# 训练好的生成器 checkpoint 路径
CHECKPOINT_PATH = "./checkpoints_pix2pix_6_4/pix2pix_epoch_0200.pth"

# 输入 mask 文件夹（用于条件生成的 mask）
MASK_DIR = r"D:\桌面\面波压制\深层数据面波压制\masks"

# 生成结果输出路径
OUTPUT_GATHER_DIR = "./generated/gathers"   # 生成的炮集
OUTPUT_MASK_DIR   = "./generated/masks"     # 对应的 mask（直接复制，保证配对）

# 网络输入尺寸（必须与训练时一致）
TARGET_SHAPE = (512, 256)  # (time, trace)

# 每张 mask 生成几个不同的炮集
# （通过 Dropout 层在推理时保持随机性实现多样性）
N_SAMPLES_PER_MASK = 1

# 是否保存预览图（PNG）
SAVE_PREVIEW = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================================

# 绘图配置（与原始代码保持一致）
rcParams['font.family'] = 'Times New Roman'
PLOT_CONFIG = {
    "dpi"      : 300,
    "fontsize" : 18,
    "figsize"  : (12, 5),
}


def load_generator(checkpoint_path, device):
    G = build_unet_generator(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    print(f"[生成器] 加载成功: {checkpoint_path}  (Epoch {ckpt.get('epoch', '?')})")
    return G


def enable_dropout(model):
    """
    推理时保持 Dropout 激活，使每次前向传播结果不同，
    从而从同一 mask 生成多样化的炮集样本（Monte Carlo Dropout）。
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def load_mask(mask_path, target_shape):
    """加载并预处理 mask，返回归一化 Tensor"""
    mask = np.load(mask_path).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[0]

    # Resize
    h, w = mask.shape
    th, tw = target_shape
    if (h, w) != (th, tw):
        mask = zoom(mask, (th / h, tw / w), order=0)  # 二值 mask 用最近邻

    # 归一化：0→-1，1→1
    mask = (mask > 0.5).astype(np.float32) * 2.0 - 1.0
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def denormalize_gather(tensor):
    """将生成器输出从 [-1,1] 还原到合理的振幅范围"""
    arr = tensor.squeeze().cpu().numpy()  # (H, W)，值域约 [-1, 1]
    return arr


def save_preview(mask_arr, gather_arr, save_path, sample_idx):
    """保存 mask 与生成炮集的对比预览图"""
    cfg = PLOT_CONFIG
    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])

    # 左：mask
    axes[0].imshow(mask_arr, aspect='auto', cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Input Mask", fontsize=cfg["fontsize"])
    axes[0].set_xlabel("Trace", fontsize=cfg["fontsize"])
    axes[0].set_ylabel("Time sample", fontsize=cfg["fontsize"])
    axes[0].tick_params(labelsize=cfg["fontsize"])

    # 右：生成炮集
    std = np.std(gather_arr)
    vmax = 3 * std if std > 0 else 1.0
    axes[1].imshow(gather_arr, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Generated Gather (Sample {sample_idx})", fontsize=cfg["fontsize"])
    axes[1].set_xlabel("Trace", fontsize=cfg["fontsize"])
    axes[1].tick_params(labelsize=cfg["fontsize"])

    plt.tight_layout()
    fig.savefig(save_path, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(fig)


def generate():
    os.makedirs(OUTPUT_GATHER_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR,   exist_ok=True)

    # 加载生成器
    G = load_generator(CHECKPOINT_PATH, DEVICE)
    G.eval()
    enable_dropout(G)   # 保持 Dropout 激活以产生多样性

    # 枚举 mask 文件
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.npy')])
    print(f"[生成] 共 {len(mask_files)} 个 mask，每个生成 {N_SAMPLES_PER_MASK} 个样本")
    print(f"       预期输出: {len(mask_files) * N_SAMPLES_PER_MASK} 对 (炮集, mask)")

    total = 0
    for mask_filename in mask_files:
        mask_path = os.path.join(MASK_DIR, mask_filename)
        mask_name = os.path.splitext(mask_filename)[0]

        # 加载 mask Tensor
        mask_tensor = load_mask(mask_path, TARGET_SHAPE).to(DEVICE)

        # 原始 mask（用于保存和可视化）
        mask_arr = (np.load(mask_path).astype(np.float32))
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
        mask_arr_binary = (mask_arr > 0.5).astype(np.float32)

        for sample_idx in range(1, N_SAMPLES_PER_MASK + 1):
            # 生成
            with torch.no_grad():
                fake_tensor = G(mask_tensor)

            gather_arr = denormalize_gather(fake_tensor)

            # 输出文件名：{原mask名}_gen{序号}
            out_stem = f"{mask_name}_gen{sample_idx:02d}"

            # 保存生成炮集
            gather_save_path = os.path.join(OUTPUT_GATHER_DIR, f"{out_stem}.npy")
            np.save(gather_save_path, gather_arr)

            # 保存对应 mask（resize 回生成尺寸，与炮集空间对齐）
            from scipy.ndimage import zoom as _zoom
            th, tw = TARGET_SHAPE
            mh, mw = mask_arr_binary.shape
            if (mh, mw) != (th, tw):
                mask_save = _zoom(mask_arr_binary, (th / mh, tw / mw), order=0)
            else:
                mask_save = mask_arr_binary
            mask_save_path = os.path.join(OUTPUT_MASK_DIR, f"{out_stem}.npy")
            np.save(mask_save_path, mask_save)

            # 保存预览图
            if SAVE_PREVIEW:
                preview_dir  = os.path.join(OUTPUT_GATHER_DIR, "previews")
                os.makedirs(preview_dir, exist_ok=True)
                preview_path = os.path.join(preview_dir, f"{out_stem}_preview.png")
                save_preview(mask_save, gather_arr, preview_path, sample_idx)

            total += 1

        print(f"  [{mask_filename}] → {N_SAMPLES_PER_MASK} 个样本生成完毕")

    print(f"\n生成完成！共输出 {total} 对样本")
    print(f"  炮集保存于: {OUTPUT_GATHER_DIR}")
    print(f"  Mask  保存于: {OUTPUT_MASK_DIR}")
    print(f"\n将以上两个文件夹的内容合并到原有训练数据中，重新训练面波识别网络即可。")


if __name__ == "__main__":
    generate()