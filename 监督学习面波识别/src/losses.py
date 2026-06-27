"""
物理约束损失函数
从 train_velocity.py 中提取，逻辑完全保持不变。
"""
import torch
import torch.nn.functional as F


def continuity_loss(pred_prob, epsilon=1e-3):
    """
    连续性/平滑性约束（Spatial Continuity）
    基于 Charbonnier TV 损失，鼓励预测的面波区域为连续条带，
    惩罚不连续性、孤立点和锯齿状边缘。

    pred_prob: 模型输出的概率图 (B, C, T, X)，已 sigmoid
    """
    # 时间方向梯度
    grad_t = pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]
    # 空间方向（道方向）梯度
    grad_x = pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]

    # 对齐维度
    grad_t = grad_t[:, :, :, :-1]
    grad_x = grad_x[:, :, :-1, :]

    # Charbonnier 惩罚：sqrt(x^2 + epsilon^2)，比 L1/L2 更适合保持边缘
    loss_t = torch.mean(torch.sqrt(grad_t ** 2 + epsilon ** 2))
    loss_x = torch.mean(torch.sqrt(grad_x ** 2 + epsilon ** 2))

    return loss_t + loss_x


def velocity_range_loss(pred_prob, dx, dt, v_min, v_max, epsilon=1e-6):
    """
    速度范围约束损失（Velocity Range Constraint）
    约束预测面波区域的局部同相轴斜率对应的速度在 [v_min, v_max] 范围内。

    物理原理：
        在炮集记录 (t-x) 中，面波同相轴斜率 slope = dt_pixel / dx_pixel，
        对应物理速度 v = dx / (slope * dt)。
        因此面波在图像上的合理斜率范围为：
            slope_min = dx / (v_max * dt)   （高速面波，斜率小）
            slope_max = dx / (v_min * dt)   （低速面波，斜率大）

    pred_prob: 模型输出的概率图 (B, C, T, X)，已 sigmoid
    dx: 道间距（米）
    dt: 采样间隔（秒）
    v_min, v_max: 面波合理速度范围（m/s）
    """
    # 对概率图做轻微平滑，使梯度更稳定
    pred_smooth = F.avg_pool2d(pred_prob, kernel_size=3, stride=1, padding=1)

    # 计算平滑后的梯度
    g_t = pred_smooth[:, :, 1:, :] - pred_smooth[:, :, :-1, :]  # 时间方向
    g_x = pred_smooth[:, :, :, 1:] - pred_smooth[:, :, :, :-1]  # 道方向

    # 对齐维度 -> (B, C, T-1, X-1)
    g_t = g_t[:, :, :, :-1]
    g_x = g_x[:, :, :-1, :]

    # 局部斜率（像素/像素）：|dt/dx|
    slope = torch.abs(g_t) / (torch.abs(g_x) + epsilon)

    # 速度范围对应的斜率边界
    # v = dx / (slope * dt)  =>  slope = dx / (v * dt)
    slope_min = dx / (v_max * dt)   # 高速对应小斜率
    slope_max = dx / (v_min * dt)   # 低速对应大斜率

    # 只在预测为面波的区域（概率 > 0.5）且 x 方向梯度显著处进行约束
    # 避免除以 0 和背景区域的干扰
    valid_mask = (pred_prob[:, :, :-1, :-1] > 0.5) & (torch.abs(g_x) > 0.01)

    # 惩罚：
    #   slope < slope_min：速度太高（像水平层位），不像面波
    #   slope > slope_max：速度太低（像陡倾噪声），不像面波
    penalty_low = F.relu(slope_min - slope)   # 速度过高惩罚
    penalty_high = F.relu(slope - slope_max)  # 速度过低惩罚

    penalty = penalty_low + penalty_high

    if valid_mask.sum() > 0:
        loss = (penalty * valid_mask.float()).sum() / valid_mask.sum()
    else:
        loss = torch.tensor(0.0, device=pred_prob.device)

    return loss
