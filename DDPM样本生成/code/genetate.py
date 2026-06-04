# diffusion_generate.py
"""
使用训练好的条件扩散模型，从 mask 批量生成含面波炮集。
采用 DDIM 采样（50步），速度快且质量高。
每个 mask 可通过不同的初始噪声生成多样化样本。
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import zoom

from model import DiffusionUNet, GaussianDiffusion

# ==================== 用户配置区 ====================
CHECKPOINT_PATH = "./checkpoints_diffusion/diffusion_epoch_0200.pth"

MASK_DIR = r"D:\桌面\面波压制\深层数据面波压制\masks"
OUTPUT_GATHER_DIR = "./generated_diffusion/gathers"
OUTPUT_MASK_DIR = "./generated_diffusion/masks"

# 必须与训练时一致
TARGET_SHAPE = (512, 256)
DDIM_STEPS = 50   # 50步即可，远快于 1000 步

# 每个 mask 生成几个不同样本（通过不同随机噪声实现多样性）
N_SAMPLES_PER_MASK = 5

SAVE_PREVIEW = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================================

rcParams['font.family'] = 'Times New Roman'
PLOT_CONFIG = {"dpi": 300, "fontsize": 18, "figsize": (12, 5)}


def load_model(checkpoint_path, device):
    model = DiffusionUNet(in_channels=2, out_channels=1, base_ch=64).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[模型] 加载成功: {checkpoint_path} (Epoch {ckpt.get('epoch', '?')})")
    return model


def load_mask(mask_path, target_shape):
    mask = np.load(mask_path).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[0]

    h, w = mask.shape
    th, tw = target_shape
    if (h, w) != (th, tw):
        mask = zoom(mask, (th / h, tw / w), order=0)

    mask = (mask > 0.5).astype(np.float32) * 2.0 - 1.0
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def save_preview(mask_arr, gather_arr, save_path, sample_idx):
    cfg = PLOT_CONFIG
    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])

    axes[0].imshow(mask_arr, aspect='auto', cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Input Mask", fontsize=cfg["fontsize"])
    axes[0].set_xlabel("Trace", fontsize=cfg["fontsize"])
    axes[0].set_ylabel("Time sample", fontsize=cfg["fontsize"])
    axes[0].tick_params(labelsize=cfg["fontsize"])

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
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, DEVICE)
    diffusion = GaussianDiffusion(timesteps=1000, beta_schedule='cosine')

    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.npy')])
    print(f"[生成] 共 {len(mask_files)} 个 mask，每个生成 {N_SAMPLES_PER_MASK} 个样本")
    print(f"       预期输出: {len(mask_files) * N_SAMPLES_PER_MASK} 对")

    total = 0
    for mask_filename in mask_files:
        mask_path = os.path.join(MASK_DIR, mask_filename)
        mask_name = os.path.splitext(mask_filename)[0]

        mask_tensor = load_mask(mask_path, TARGET_SHAPE).to(DEVICE)

        # 原始 mask 数组（用于保存和预览）
        mask_arr = np.load(mask_path).astype(np.float32)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
        mask_arr_binary = (mask_arr > 0.5).astype(np.float32)

        for sample_idx in range(1, N_SAMPLES_PER_MASK + 1):
            # 每次生成使用不同的随机种子，得到多样性
            torch.manual_seed(total + 42)

            with torch.no_grad():
                fake = diffusion.ddim_sample(
                    model,
                    shape=(1, 1, TARGET_SHAPE[0], TARGET_SHAPE[1]),
                    cond=mask_tensor,
                    device=DEVICE,
                    ddim_timesteps=DDIM_STEPS,
                    eta=0.0
                )

            gather_arr = fake.squeeze().cpu().numpy()

            out_stem = f"{mask_name}_gen{sample_idx:02d}"

            # 保存生成炮集
            np.save(os.path.join(OUTPUT_GATHER_DIR, f"{out_stem}.npy"), gather_arr)

            # 保存配对 mask（resize 对齐）
            mh, mw = mask_arr_binary.shape
            if (mh, mw) != TARGET_SHAPE:
                mask_save = zoom(mask_arr_binary, (TARGET_SHAPE[0] / mh, TARGET_SHAPE[1] / mw), order=0)
            else:
                mask_save = mask_arr_binary
            np.save(os.path.join(OUTPUT_MASK_DIR, f"{out_stem}.npy"), mask_save)

            # 预览图
            if SAVE_PREVIEW:
                preview_dir = os.path.join(OUTPUT_GATHER_DIR, "previews")
                os.makedirs(preview_dir, exist_ok=True)
                save_preview(mask_save, gather_arr,
                             os.path.join(preview_dir, f"{out_stem}_preview.png"), sample_idx)

            total += 1

        print(f"  [{mask_filename}] → {N_SAMPLES_PER_MASK} 个样本生成完毕")

    print(f"\n生成完成！共输出 {total} 对样本")
    print(f"  炮集: {OUTPUT_GATHER_DIR}")
    print(f"  Mask: {OUTPUT_MASK_DIR}")
    print(f"\n将以上数据合并到原有训练集，重新训练面波识别网络即可。")


if __name__ == "__main__":
    generate()