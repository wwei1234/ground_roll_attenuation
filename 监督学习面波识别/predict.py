# predict.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rcParams

from src.utils import ensure_dir, map_label_filename, load_model

# ============================================================
#  ★ 绘图配置（参考标准绘图代码）
# ============================================================
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False

PLOT_CONFIG = {
    "dpi"           : 600,
    "formats"       : ["png", "eps", "pdf"],
    "fontsize"      : 24,
    "figsize_single": (6, 8),
    "figsize_compare": (22, 5),
    "figsize_colorbar_original": (0.5, 4),
    "figsize_colorbar_prediction": (0.5, 4),
}

# ==================== 用户配置区（请在此处修改）====================
# 模型路径
MODEL_PATH = "./checkpoints/model_best.pth"
MODEL_FILE = r"U_Net_CBAM.py"
MODEL_CLASS_NAME = "UNet"

# 数据路径
DATA_DIR = r"D:\桌面\面波压制\深层数据面波压制\原始炮集"
LABEL_DIR = r"D:\桌面\面波压制\深层数据面波压制\masks"
# LABEL_DIR = None  # 取消这行注释可关闭标签对比

# 输出路径
OUTPUT_DIR = "./predictions_postprocessed"

# 预测设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
THRESHOLD = 0.4

# 指定要预测的文件（为空则预测全部）
PREDICT_FILES = ["real_data_gather185.npy", "real_data_gather182.npy", 
                 "real_data_gather183.npy", "real_data_gather184.npy"]

# ==================== 后处理约束参数（新增）====================
ENABLE_POSTPROCESS = True       # 总开关：是否启用后处理约束

# 1. 概率图平滑（消除孤立的噪声响应）
PROB_SMOOTH_SIGMA = 1.0         # 高斯平滑核 sigma，越大越平滑（0 关闭）

# 2. 形态学开运算（先腐蚀去除孤立点，再膨胀恢复主体）
MORPH_OPEN_KERNEL = 3           # 开运算结构元素大小（像素），0 关闭

# 3. 连通域面积过滤（面波是连续条带，孤立点面积很小）
MIN_AREA = 100                  # 最小保留面积（像素数），小于此值的连通域被剔除

# 4. 速度范围后验过滤（可选，默认关闭）
# 原理：预测出的每个连通域计算其局部速度，不在合理范围内的视为误检
ENABLE_VELOCITY_FILTER = False  # 是否启用速度后验过滤
DX = 10.0                       # 道间距（米）
DT = 0.004                      # 采样间隔（秒）
V_MIN = 100.0                   # 最小面波速度（m/s）
V_MAX = 1000.0                  # 最大面波速度（m/s）

# 5. 测试时增强（TTA）：左右翻转预测再平均，减少随机噪声
ENABLE_TTA = True               # 是否启用 TTA
# ================================================================


def predict_file(model, data_path, device, threshold=0.5, use_tta=False):
    """
    对单炮数据进行预测
    use_tta: 是否使用测试时增强（左右翻转平均）
    """
    data = np.load(data_path).astype(np.float32)
    original_shape = data.shape
    
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 3:
        pass
    else:
        raise ValueError(f"不支持的数据维度: {data.ndim}")
    
    data_tensor = torch.from_numpy(data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 正常预测
        output = model(data_tensor)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # TTA：左右翻转后预测，再翻转回来平均
        if use_tta:
            data_flipped = torch.flip(data_tensor, dims=[-1])
            output_flipped = model(data_flipped)
            pred_prob_flipped = torch.sigmoid(output_flipped).squeeze().cpu().numpy()
            pred_prob_flipped = np.flip(pred_prob_flipped, axis=-1)
            # 平均两个预测
            pred_prob = (pred_prob + pred_prob_flipped) / 2.0
    
    pred_binary = (pred_prob > threshold).astype(np.float32)
    
    return pred_prob, pred_binary, original_shape


def postprocess_prediction(pred_prob, pred_binary, threshold=0.5,
                           smooth_sigma=1.0, morph_kernel=3, min_area=100,
                           enable_velocity_filter=False,
                           dx=10.0, dt=0.004, v_min=100.0, v_max=1000.0):
    """
    后处理约束：消除面波范围外的孤立误检点
    
    处理流程：
        1. 高斯平滑概率图 -> 消除孤立的噪声响应
        2. 重新二值化
        3. 形态学开运算 -> 去除小面积孤立点
        4. 连通域面积过滤 -> 只保留大面积连续条带（面波）
        5. 速度范围后验过滤（可选）-> 剔除速度不合理的连通域
        6. 概率图与二值图对齐 -> 被剔除区域概率置 0
    """
    from scipy import ndimage
    
    # 1. 概率图高斯平滑
    if smooth_sigma > 0:
        pred_prob = ndimage.gaussian_filter(pred_prob, sigma=smooth_sigma)
    
    # 2. 重新二值化
    pred_binary = (pred_prob > threshold).astype(np.float32)
    
    # 3. 形态学开运算（先腐蚀去噪点，再膨胀恢复主体）
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
        eroded = ndimage.binary_erosion(pred_binary, structure=kernel).astype(np.float32)
        opened = ndimage.binary_dilation(eroded, structure=kernel).astype(np.float32)
        pred_binary = opened
    
    # 4. 连通域面积过滤：只保留面积 >= min_area 的连通域
    labeled, num_features = ndimage.label(pred_binary)
    if num_features > 0:
        sizes = ndimage.sum(pred_binary, labeled, range(1, num_features + 1))
        mask = np.zeros_like(pred_binary)
        for i, size in enumerate(sizes, 1):
            if size >= min_area:
                mask[labeled == i] = 1
        pred_binary = mask
    
    # 5. 速度范围后验过滤（可选）
    if enable_velocity_filter and v_min > 0 and v_max > 0:
        labeled, num_features = ndimage.label(pred_binary)
        if num_features > 0:
            mask = np.zeros_like(pred_binary)
            for i in range(1, num_features + 1):
                component = (labeled == i).astype(np.float32)
                coords = np.argwhere(component > 0)
                if len(coords) > 1:
                    t_range = coords[:, 0].max() - coords[:, 0].min()
                    x_range = coords[:, 1].max() - coords[:, 1].min()
                    if x_range > 0:
                        slope = t_range / x_range  # pixel / pixel
                        v = dx / (slope * dt + 1e-9)
                        if v_min <= v <= v_max:
                            mask[labeled == i] = 1
            pred_binary = mask
    
    # 6. 概率图对齐：被后处理剔除的区域概率置 0
    pred_prob = pred_prob * pred_binary
    
    return pred_prob, pred_binary


def save_prediction_plot(data_path, pred_prob, pred_binary, save_dir, label=None):
    """保存4个独立的单图：原始数据、预测概率、预测二值、标签"""
    data = np.load(data_path)
    if data.ndim == 3:
        data = data[0]
    
    cfg = PLOT_CONFIG
    time_max_s = 3.0
    extent = [0, data.shape[1], time_max_s, 0]
    
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    
    vmin_data = -np.std(data)*3
    vmax_data = np.std(data)*3
    
    # 1. 原始数据
    fig, ax = plt.subplots(figsize=cfg["figsize_single"])
    ax.imshow(data, aspect='auto', cmap='seismic', vmin=vmin_data, vmax=vmax_data, extent=extent)
    ax.set_xlabel("Trace", fontsize=cfg["fontsize"])
    ax.set_ylabel("Time (s)", fontsize=cfg["fontsize"])
    ax.tick_params(labelsize=cfg["fontsize"])
    plt.tight_layout()
    for fmt in cfg["formats"]:
        fig.savefig(os.path.join(save_dir, f"{base_name}_original.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(fig)
    
    cbar_fig, cbar_ax = plt.subplots(figsize=cfg["figsize_colorbar_original"])
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin_data, vmax=vmax_data))
    sm.set_array([])
    cbar = cbar_fig.colorbar(sm, cax=cbar_ax, label="Amplitude")
    cbar_ax.tick_params(labelsize=cfg["fontsize"])
    cbar.set_label("Amplitude", fontsize=cfg["fontsize"])
    for fmt in cfg["formats"]:
        cbar_fig.savefig(os.path.join(save_dir, f"{base_name}_original_colorbar.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(cbar_fig)
    
    # 2. 预测概率
    fig, ax = plt.subplots(figsize=cfg["figsize_single"])
    ax.imshow(pred_prob, aspect='auto', cmap='hot', vmin=0, vmax=1, extent=extent)
    ax.set_xlabel("Trace", fontsize=cfg["fontsize"])
    ax.set_ylabel("Time (s)", fontsize=cfg["fontsize"])
    ax.tick_params(labelsize=cfg["fontsize"])
    plt.tight_layout()
    for fmt in cfg["formats"]:
        fig.savefig(os.path.join(save_dir, f"{base_name}_probability.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(fig)
    
    cbar_fig, cbar_ax = plt.subplots(figsize=cfg["figsize_colorbar_prediction"])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = cbar_fig.colorbar(sm, cax=cbar_ax, label="Probability")
    cbar_ax.tick_params(labelsize=cfg["fontsize"])
    cbar.set_label("Probability", fontsize=cfg["fontsize"])
    for fmt in cfg["formats"]:
        cbar_fig.savefig(os.path.join(save_dir, f"{base_name}_probability_colorbar.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(cbar_fig)
    
    # 3. 预测二值
    fig, ax = plt.subplots(figsize=cfg["figsize_single"])
    ax.imshow(pred_binary, aspect='auto', cmap='gray', vmin=0, vmax=1, extent=extent)
    ax.set_xlabel("Trace", fontsize=cfg["fontsize"])
    ax.set_ylabel("Time (s)", fontsize=cfg["fontsize"])
    ax.tick_params(labelsize=cfg["fontsize"])
    plt.tight_layout()
    for fmt in cfg["formats"]:
        fig.savefig(os.path.join(save_dir, f"{base_name}_binary.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(fig)
    
    cbar_fig, cbar_ax = plt.subplots(figsize=cfg["figsize_colorbar_prediction"])
    sm = plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = cbar_fig.colorbar(sm, cax=cbar_ax, label="Binary")
    cbar_ax.tick_params(labelsize=cfg["fontsize"])
    cbar.set_label("Binary", fontsize=cfg["fontsize"])
    for fmt in cfg["formats"]:
        cbar_fig.savefig(os.path.join(save_dir, f"{base_name}_binary_colorbar.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(cbar_fig)
    
    # 4. 标签（如果有）
    if label is not None:
        if label.ndim == 3:
            label = label[0]
        fig, ax = plt.subplots(figsize=cfg["figsize_single"])
        ax.imshow(label, aspect='auto', cmap='gray', vmin=0, vmax=1, extent=extent)
        ax.set_xlabel("Trace", fontsize=cfg["fontsize"])
        ax.set_ylabel("Time (s)", fontsize=cfg["fontsize"])
        ax.tick_params(labelsize=cfg["fontsize"])
        plt.tight_layout()
        for fmt in cfg["formats"]:
            fig.savefig(os.path.join(save_dir, f"{base_name}_label.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
        plt.close(fig)
        
        cbar_fig, cbar_ax = plt.subplots(figsize=cfg["figsize_colorbar_prediction"])
        sm = plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = cbar_fig.colorbar(sm, cax=cbar_ax, label="Label")
        cbar_ax.tick_params(labelsize=cfg["fontsize"])
        cbar.set_label("Label", fontsize=cfg["fontsize"])
        for fmt in cfg["formats"]:
            cbar_fig.savefig(os.path.join(save_dir, f"{base_name}_label_colorbar.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
        plt.close(cbar_fig)


def save_prediction_overlay(data_path, pred_binary, save_dir):
    """预测面波位置在原始数据上的投影"""
    data = np.load(data_path)
    if data.ndim == 3:
        data = data[0]
    
    cfg = PLOT_CONFIG
    time_max_s = 3.0
    extent = [0, data.shape[1], time_max_s, 0]
    
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    
    fig, ax = plt.subplots(1, 1, figsize=cfg["figsize_single"])
    ax.imshow(data, aspect='auto', cmap='seismic', vmin=-np.std(data)*3, vmax=np.std(data)*3, extent=extent)
    ax.imshow(pred_binary, aspect='auto', cmap='Reds', alpha=0.4, vmin=0, vmax=1, extent=extent)
    ax.set_xlabel("Trace", fontsize=cfg["fontsize"])
    ax.set_ylabel("Time (s)", fontsize=cfg["fontsize"])
    ax.tick_params(labelsize=cfg["fontsize"])
    plt.tight_layout()
    
    for fmt in cfg["formats"]:
        fig.savefig(os.path.join(save_dir, f"{base_name}_overlay.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(fig)
    
    cbar_fig, cbar_ax = plt.subplots(figsize=cfg["figsize_colorbar_original"])
    sm0 = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=-np.std(data)*3, vmax=np.std(data)*3))
    sm0.set_array([])
    cbar = cbar_fig.colorbar(sm0, cax=cbar_ax, label="Amplitude")
    cbar_ax.tick_params(labelsize=cfg["fontsize"])
    cbar.set_label("Amplitude", fontsize=cfg["fontsize"])
    for fmt in cfg["formats"]:
        cbar_fig.savefig(os.path.join(save_dir, f"{base_name}_overlay_colorbar_seismic.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(cbar_fig)
    
    cbar_fig, cbar_ax = plt.subplots(figsize=cfg["figsize_colorbar_prediction"])
    sm1 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1))
    sm1.set_array([])
    cbar = cbar_fig.colorbar(sm1, cax=cbar_ax, label="Prediction")
    cbar_ax.tick_params(labelsize=cfg["fontsize"])
    cbar.set_label("Prediction", fontsize=cfg["fontsize"])
    for fmt in cfg["formats"]:
        cbar_fig.savefig(os.path.join(save_dir, f"{base_name}_overlay_colorbar_pred.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(cbar_fig)


def save_difference_map(data_path, pred_binary, label, save_dir):
    """预测面波位置和标签面波位置的差异图"""
    data = np.load(data_path)
    if data.ndim == 3:
        data = data[0]
    if label.ndim == 3:
        label = label[0]

    cfg = PLOT_CONFIG
    time_max_s = 3.0
    extent = [0, data.shape[1], time_max_s, 0]
    
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(data_path))[0]

    difference = np.zeros((*pred_binary.shape, 3), dtype=np.float32)
    difference[..., 0] = pred_binary
    difference[..., 1] = label
    
    fig, ax = plt.subplots(figsize=cfg["figsize_single"])
    ax.imshow(difference, aspect='auto', extent=extent)
    ax.set_xlabel("Trace", fontsize=cfg["fontsize"])
    ax.set_ylabel("Time (s)", fontsize=cfg["fontsize"])
    ax.tick_params(labelsize=cfg["fontsize"])
    plt.tight_layout()
    
    for fmt in cfg["formats"]:
        fig.savefig(os.path.join(save_dir, f"{base_name}_difference.{fmt}"), format=fmt, dpi=cfg["dpi"], bbox_inches='tight')
    plt.close(fig)


def calculate_metrics(pred_binary, label):
    """计算预测准确率相关指标"""
    if pred_binary.ndim == 3:
        pred_binary = pred_binary[0]
    if label.ndim == 3:
        label = label[0]
    
    pred_binary = (pred_binary > 0.5).astype(np.float32)
    label = (label > 0.5).astype(np.float32)
    
    tp = np.sum(pred_binary * label)
    tn = np.sum((1 - pred_binary) * (1 - label))
    fp = np.sum(pred_binary * (1 - label))
    fn = np.sum((1 - pred_binary) * label)
    
    epsilon = 1e-7
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'dice': float(dice),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def main():
    ensure_dir(OUTPUT_DIR)
    
    # 加载模型
    model = load_model(MODEL_PATH, MODEL_FILE, MODEL_CLASS_NAME, DEVICE)
    
    # 获取预测文件列表
    if len(PREDICT_FILES) > 0:
        predict_files = PREDICT_FILES
    else:
        predict_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
    
    print(f"待预测文件数: {len(predict_files)}")
    if ENABLE_POSTPROCESS:
        print(f"后处理已启用: 平滑sigma={PROB_SMOOTH_SIGMA}, 开运算核={MORPH_OPEN_KERNEL}, "
              f"最小面积={MIN_AREA}, TTA={ENABLE_TTA}")
    
    all_metrics = []
    
    for filename in tqdm(predict_files, desc="Predicting"):
        data_path = os.path.join(DATA_DIR, filename)
        
        # 预测（可选 TTA）
        pred_prob, pred_binary, original_shape = predict_file(
            model, data_path, DEVICE, THRESHOLD, use_tta=ENABLE_TTA
        )
        
        # 后处理约束（消除孤立误检点）
        if ENABLE_POSTPROCESS:
            pred_prob, pred_binary = postprocess_prediction(
                pred_prob, pred_binary, threshold=THRESHOLD,
                smooth_sigma=PROB_SMOOTH_SIGMA,
                morph_kernel=MORPH_OPEN_KERNEL,
                min_area=MIN_AREA,
                enable_velocity_filter=ENABLE_VELOCITY_FILTER,
                dx=DX, dt=DT, v_min=V_MIN, v_max=V_MAX
            )
        
        # 按文件名创建输出目录
        name = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(OUTPUT_DIR, name)
        ensure_dir(file_output_dir)
        
        # 保存预测数据
        np.save(os.path.join(file_output_dir, f"{name}_pred_prob.npy"), pred_prob)
        np.save(os.path.join(file_output_dir, f"{name}_pred_binary.npy"), pred_binary)
        
        # 加载标签
        label = None
        if LABEL_DIR is not None:
            label_path = os.path.join(LABEL_DIR, filename)
            if not os.path.exists(label_path):
                alt_label_name = map_label_filename(filename)
                alt_label_path = os.path.join(LABEL_DIR, alt_label_name)
                if os.path.exists(alt_label_path):
                    label_path = alt_label_path
            if os.path.exists(label_path):
                label = np.load(label_path)
            else:
                print(f"警告: 未找到标签文件: {filename}")
        
        # 保存图片
        save_prediction_plot(data_path, pred_prob, pred_binary, file_output_dir, label)
        save_prediction_overlay(data_path, pred_binary, file_output_dir)
        
        if label is not None:
            save_difference_map(data_path, pred_binary, label, file_output_dir)
            metrics = calculate_metrics(pred_binary, label)
            metrics['filename'] = filename
            metrics['output_dir'] = file_output_dir
            all_metrics.append(metrics)
    
    # 保存评估指标
    if all_metrics:
        import csv
        metrics_csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n评估指标已保存: {metrics_csv_path}")
        
        print("\n平均评估指标:")
        avg_metrics = {
            'Accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'Precision': np.mean([m['precision'] for m in all_metrics]),
            'Recall': np.mean([m['recall'] for m in all_metrics]),
            'F1 Score': np.mean([m['f1'] for m in all_metrics]),
            'IoU': np.mean([m['iou'] for m in all_metrics]),
            'Dice': np.mean([m['dice'] for m in all_metrics]),
        }
        for metric_name, value in avg_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print(f"\n预测完成！结果保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()