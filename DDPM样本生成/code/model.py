# diffusion_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 时间步嵌入 ====================
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ==================== FiLM 调制 ResBlock ====================
class ResBlock(nn.Module):
    """
    支持时间步 FiLM 调制的残差块。
    将时间嵌入映射为 scale/shift，对特征做逐通道缩放+平移。
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)  # scale + shift
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM: (B, out_ch*2) -> scale/shift
        t = self.time_mlp(F.silu(t_emb))
        t = t[:, :, None, None]
        scale, shift = t.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ==================== 轻量注意力（仅在 16x16 分辨率附近使用） ====================
class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q * (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhdi,bhdj->bhij', q, k), dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        return x + out


# ==================== 条件扩散 U-Net ====================
class DiffusionUNet(nn.Module):
    """
    输入: [noisy_gather, mask] 沿通道拼接 -> (B, 2, H, W)
    输出: 预测的噪声 -> (B, 1, H, W)
    
    架构: 4-level Encoder-Decoder + Skip Connections + Time Embedding
    下采样路径: 512x256 -> 256x128 -> 128x64 -> 64x32
    """
    def __init__(self, in_channels=2, out_channels=1, base_ch=64,
                 ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 time_emb_dim=256, dropout=0.1):
        super().__init__()

        # 时间步嵌入 MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # 下采样
        self.downs = nn.ModuleList()
        ch_list = [base_ch]
        now_ch = base_ch

        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(now_ch, out_ch, time_emb_dim, dropout))
                now_ch = out_ch
                ch_list.append(now_ch)

            if i != len(ch_mult) - 1:  # 最后层不下采样
                self.downs.append(nn.Conv2d(now_ch, now_ch, 3, stride=2, padding=1))
                ch_list.append(now_ch)

        # 中间层（瓶颈）
        self.mid1 = ResBlock(now_ch, now_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(now_ch)
        self.mid2 = ResBlock(now_ch, now_ch, time_emb_dim, dropout)

        # 上采样
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(now_ch + ch_list.pop(), out_ch, time_emb_dim, dropout))
                now_ch = out_ch

            if i != len(ch_mult) - 1:
                self.ups.append(nn.ConvTranspose2d(now_ch, now_ch, 4, stride=2, padding=1))

        # 输出
        self.out_norm = nn.GroupNorm(8, now_ch)
        self.out_conv = nn.Conv2d(now_ch, out_channels, 3, padding=1)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        x: (B, C, H, W)  -- 已拼接 [noisy_gather, mask]
        t: (B,)          -- 时间步
        """
        t_emb = self.time_embed(t)

        h = self.input_conv(x)
        hs = [h]

        # Down
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
            hs.append(h)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Up
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


# ==================== 高斯扩散过程（DDPM + DDIM） ====================
class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule='cosine',
                 beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            # Improved DDPM 的 cosine schedule，通常比 linear 更稳定
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """前向加噪: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, cond, t, noise=None,
                 lambda_l1=0.0, lambda_freq=0.0, lambda_trace=0.0):
        """
        训练损失：噪声预测的 MSE，加上可选的 x0 物理约束损失。
        cond: mask (B,1,H,W)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        x_input = torch.cat([x_noisy, cond], dim=1)  # 条件拼接

        predicted_noise = model(x_input, t)
        loss = F.mse_loss(predicted_noise, noise)

        # 混合损失：从预测噪声恢复 x0，施加物理约束（与 Pix2Pix 中的损失一致）
        if lambda_l1 > 0 or lambda_freq > 0 or lambda_trace > 0:
            sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            pred_x0 = (x_noisy - sqrt_one_minus_alpha_t * predicted_noise) / (sqrt_alpha_t + 1e-8)

            if lambda_l1 > 0:
                loss = loss + lambda_l1 * F.l1_loss(pred_x0, x_start)
            if lambda_freq > 0:
                loss = loss + lambda_freq * self._frequency_loss(pred_x0, x_start)
            if lambda_trace > 0:
                loss = loss + lambda_trace * self._trace_continuity_loss(pred_x0, x_start)

        return loss

    @staticmethod
    def _frequency_loss(real, fake):
        def low_freq_amp(x):
            spec = torch.fft.rfft(x, dim=-2)
            amp = torch.abs(spec)
            n_low = max(1, amp.shape[-2] // 5)
            return amp[..., :n_low, :]
        return F.l1_loss(low_freq_amp(fake), low_freq_amp(real))

    @staticmethod
    def _trace_continuity_loss(fake, real):
        diff_real = real[..., 1:] - real[..., :-1]
        diff_fake = fake[..., 1:] - fake[..., :-1]
        return F.l1_loss(diff_fake, diff_real)

    @torch.no_grad()
    def p_sample(self, model, x, cond, t):
        """DDPM 单步去噪（训练时验证用，推理推荐 DDIM）"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        x_input = torch.cat([x, cond], dim=1)
        predicted_noise = model(x_input, t)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, shape, cond, device):
        """DDPM 完整采样（1000步，慢，仅作参考）"""
        b = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, cond, t)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, device, ddim_timesteps=50, eta=0.0):
        """
        DDIM 快速采样（推荐）。
        ddim_timesteps=50: 仅需 50 步即可生成高质量样本。
        eta=0: 完全确定性，同一 mask + 同一随机种子 -> 相同样本。
        """
        # 构造均匀子序列，例如 1000 步中取 50 步: [0, 20, 40, ..., 980]
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = list(range(0, self.timesteps, c))

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(len(ddim_timestep_seq))):
            t = torch.full((b,), ddim_timestep_seq[i], device=device, dtype=torch.long)

            x_input = torch.cat([x, cond], dim=1)
            predicted_noise = model(x_input, t)

            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)

            if i > 0:
                prev_t = torch.full((b,), ddim_timestep_seq[i - 1], device=device, dtype=torch.long)
                alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, x.shape)
            else:
                alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)

            # 预测 x0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

            # 方向项
            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * predicted_noise

            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
            if eta > 0 and i > 0:
                x = x + sigma_t * torch.randn_like(x)

        return x