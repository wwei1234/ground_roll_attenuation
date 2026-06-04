from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道注意力模块 (Channel Attention Module)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块 (Spatial Attention Module)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM模块 (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x

# 带有CBAM的双卷积类
class DoubleConvWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConvWithCBAM, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 添加CBAM注意力机制
        self.cbam = CBAM(out_channels)
    
    def forward(self, x):
        x = self.double_conv(x)
        x = self.cbam(x)
        return x

# 原始的双卷积类（编码器第一层不使用CBAM）
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# 带有CBAM的下采样类
class DownWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownWithCBAM, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConvWithCBAM(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

# 带有CBAM的上采样类
class UpWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpWithCBAM, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvWithCBAM(in_channels, out_channels)
 
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
 
        # 保证x1和x2在维度上一致
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
 
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# 输出卷积类
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

# 带有CBAM注意力机制的UNet
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 编码器 - 第一层使用原始双卷积，其他层使用带CBAM的双卷积
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = DownWithCBAM(base_c, base_c * 2)
        self.down2 = DownWithCBAM(base_c * 2, base_c * 4)
        self.down3 = DownWithCBAM(base_c * 4, base_c * 8)
        self.down4 = DownWithCBAM(base_c * 8, base_c * 16)
 
        # 解码器 - 全部使用带CBAM的上采样
        self.up1 = UpWithCBAM(base_c * 16, base_c * 8)
        self.up2 = UpWithCBAM(base_c * 8, base_c * 4)
        self.up3 = UpWithCBAM(base_c * 4, base_c * 2)
        self.up4 = UpWithCBAM(base_c * 2, base_c)
        
        # 输出层
        self.out_conv = OutConv(base_c, num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器路径
        x1 = self.in_conv(x)        # [N, 64, H, W]
        x2 = self.down1(x1)         # [N, 128, H/2, W/2]
        x3 = self.down2(x2)         # [N, 256, H/4, W/4]
        x4 = self.down3(x3)         # [N, 512, H/8, W/8]
        x5 = self.down4(x4)         # [N, 1024, H/16, W/16]
        
        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)        # [N, 512, H/8, W/8]
        x = self.up2(x, x3)         # [N, 256, H/4, W/4]
        x = self.up3(x, x2)         # [N, 128, H/2, W/2]
        x = self.up4(x, x1)         # [N, 64, H, W]
        
        # 输出层
        logits = self.out_conv(x)   # [N, num_classes, H, W]
 
        return logits