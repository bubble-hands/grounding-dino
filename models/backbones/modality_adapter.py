import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


class ModalityAdapter(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.residual = nn.Conv2d(in_channels, embed_dim, kernel_size=1) if in_channels != embed_dim else nn.Identity()

    def forward(self, x):
        shortcut = self.residual(x)
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = self.act(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x + shortcut


class MultiModalFeatureFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities=3, reduction_ratio=4):
        super().__init__()
        self.num_modalities = num_modalities
        self.channel_attention = nn.Sequential(
            nn.Conv2d(embed_dim * num_modalities, embed_dim * num_modalities // reduction_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim * num_modalities // reduction_ratio, embed_dim * num_modalities, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_modalities, num_modalities, kernel_size=7, padding=3, groups=num_modalities),
            nn.GELU(),
            nn.Conv2d(num_modalities, num_modalities, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(embed_dim * num_modalities, embed_dim, kernel_size=1)

    def forward(self, modality_features):
        stacked = torch.cat(modality_features, dim=1)
        b, c, h, w = stacked.shape

        ca_weights = self.channel_attention(stacked)
        ca_out = stacked * ca_weights

        spatial_features = [f.mean(dim=1, keepdim=True) for f in modality_features]
        spatial_stacked = torch.cat(spatial_features, dim=1)
        sa_weights = self.spatial_attention(spatial_stacked)
        sa_weights = sa_weights.unsqueeze(1).expand(b, len(modality_features), -1, -1, -1)
        sa_weights = rearrange(sa_weights, 'b m c h w -> b (m c) h w')
        sa_out = ca_out * sa_weights

        return self.fusion_conv(sa_out)