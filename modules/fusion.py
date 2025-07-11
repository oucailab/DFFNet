import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, (1, 1))
        attention = self.mlp(gap)
        return x + attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()

        self.conv_spatial_map = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat_pool = torch.cat([avg_pool, max_pool], dim=1)

        spatial_attention_map = self.conv_spatial_map(concat_pool)
        return x + spatial_attention_map

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        channels_per_group = C // self.groups

        x = x.view(B, self.groups, channels_per_group, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        return x

class SpectralSpatialAdaptiveFusionBlock(nn.Module):
    def __init__(self, hsi_channels, sar_channels, out_channels, shuffle_groups=2, reduction_ratio_ca=16, kernel_size_sa=5):
        super().__init__()
        self.channel_attn_hsi = ChannelAttention(hsi_channels, reduction_ratio=reduction_ratio_ca)
        self.spatial_attn_sar = SpatialAttention(kernel_size=kernel_size_sa)

        concatenated_channels = hsi_channels + sar_channels
        self.channel_shuffle = ChannelShuffle(groups=shuffle_groups)

        self.conv_out = nn.Conv2d(concatenated_channels, out_channels, kernel_size=1, bias=False)


    def forward(self, F_h, F_x):
        F_h_prime = self.channel_attn_hsi(F_h)
        F_x_prime = self.spatial_attn_sar(F_x)

        fused_features = torch.cat([F_h_prime, F_x_prime], dim=1)

        shuffled_features = self.channel_shuffle(fused_features)

        F_o = self.conv_out(shuffled_features)

        return F_o