# model.py
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
#  生成器：U-Net（与原版相同，不改动）
# ─────────────────────────────────────────────────────────────

class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, 4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu   = nn.ReLU(inplace=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, stride=2, padding=1)
            model  = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, stride=2, padding=1, bias=False)
            upnorm = nn.BatchNorm2d(outer_nc)
            model  = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, stride=2, padding=1, bias=False)
            upnorm = nn.BatchNorm2d(outer_nc)
            model  = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


def build_unet_generator(in_channels=1, out_channels=1, ngf=64, n_down=7):
    if n_down < 5:
        raise ValueError('n_down must be at least 5')

    unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
    for _ in range(n_down - 5):
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8,
                                             submodule=unet_block, dropout=True)
    unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block)
    unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block)
    unet_block = UNetSkipConnectionBlock(ngf,     ngf * 2, submodule=unet_block)
    unet_block = UNetSkipConnectionBlock(out_channels, ngf, input_nc=in_channels,
                                         submodule=unet_block, outermost=True)
    return unet_block


# ─────────────────────────────────────────────────────────────
#  判别器：PatchGAN，各层分离以支持 Feature Matching Loss
#
#  改动说明：
#    原版用 nn.Sequential 把所有层打包，无法提取中间特征。
#    现在把各层存为独立属性（layer1~layer4 + out_layer），
#    forward 可选 return_features=True 返回各层特征列表，
#    用于计算 Feature Matching Loss。
#    外部调用方式不变：D(condition, target) 仍返回 patch 评分图。
# ─────────────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    条件 PatchGAN 判别器，支持返回中间层特征（用于 Feature Matching Loss）。
    in_channels = mask 通道数 + 炮集通道数 = 1 + 1 = 2
    """

    def __init__(self, in_channels=2, ndf=64):
        super().__init__()

        # 第一层：不加 BatchNorm
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 第二层
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 第三层
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 第四层：stride=1，扩大感受野
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 输出层：每个 patch 一个评分（无 Sigmoid，配合 MSELoss）
        self.out_layer = nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)

    def forward(self, condition, target, return_features=False):
        """
        condition      : mask，(B, 1, H, W)
        target         : 炮集（真实或生成），(B, 1, H, W)
        return_features: 为 True 时同时返回各层特征列表，用于 Feature Matching Loss

        返回：
            patch 评分图 (B, 1, h, w)
            [可选] 各层特征列表 [f1, f2, f3, f4]
        """
        x  = torch.cat([condition, target], dim=1)  # (B, 2, H, W)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = self.out_layer(f4)

        if return_features:
            return out, [f1, f2, f3, f4]
        return out


# ─────────────────────────────────────────────────────────────
#  权重初始化
# ─────────────────────────────────────────────────────────────

def init_weights(net, gain=0.02):
    def _init(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(_init)
    return net