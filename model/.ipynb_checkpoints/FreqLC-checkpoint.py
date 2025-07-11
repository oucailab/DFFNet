from modules.dct_util import *
from einops import rearrange
import numbers
import math
import torch.nn.functional as F
import yaml
from modules.Mamba_v3 import *
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba


def d4_to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def d3_to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias, mu_sigma=False):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.mu_sigma = mu_sigma
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.norm_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.norm_bias:
            x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        else:
            x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight
        if self.mu_sigma:
            return x, mu, sigma
        else:
            return x


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False, out_dir=None):
        super(LayerNorm, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm(dim, bias, mu_sigma)
        self.out_dir = out_dir

    def forward(self, x):
        h, w = x.shape[-2:]
        x = d4_to_3d(x)

        if self.mu_sigma:
            x, mu, sigma = self.body(x)
            return d3_to_4d(x, h, w), d3_to_4d(mu, h, w), d3_to_4d(sigma, h, w)
        else:
            x = self.body(x)
            return d3_to_4d(x, h, w)


class ImprovedAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        # Mamba模块
        self.mamba_ul = MambaBlock(hidden_dim=dim)
        self.mamba_ur = MambaBlock(hidden_dim=dim)
        self.mamba_ll = MambaBlock(hidden_dim=dim)
        self.mamba_lr = MambaBlock(hidden_dim=dim)

    def elastic_split(self, x):
        # 修改后的动态分割方法
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_new = (H + pad_h) // 2
        W_new = (W + pad_w) // 2
        upper = x[:, :, :H_new, :]
        lower = x[:, :, H_new:, :]
        ul = upper[:, :, :, :W_new]
        ur = upper[:, :, :, W_new:]
        ll = lower[:, :, :, :W_new]
        lr = lower[:, :, :, W_new:]
        return ul, ur, ll, lr, (pad_h, pad_w)

    def elastic_merge(self, patches, pad_size):
        """关键修复：合并分块并移除填充"""
        ul, ur, ll, lr = patches
        pad_h, pad_w = pad_size

        # 合并上下部分
        upper = torch.cat([ul, ur], dim=-1)
        lower = torch.cat([ll, lr], dim=-1)
        merged = torch.cat([upper, lower], dim=-2)

        # 移除填充
        if pad_h > 0:
            merged = merged[:, :, :-pad_h, :]
        if pad_w > 0:
            merged = merged[:, :, :, :-pad_w]

        return merged

    def forward(self, x):
        # 弹性分割
        ul, ur, ll, lr, pad_size = self.elastic_split(x)

        # 处理每个分块
        ul = self.mamba_ul(ul)
        ur = self.mamba_ur(ur)
        ll = self.mamba_ll(ll)
        lr = self.mamba_lr(lr)

        # 弹性合并
        merged = self.elastic_merge([ul, ur, ll, lr], pad_size)

        return merged


# EnhancedFreqLCBlock 类修正
class EnhancedFreqLCBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type, window_size=8, cs='dct'):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.cs = cs

        # 初始化DCT模块
        if 'nodct' in cs:
            self.dct = nn.Identity()
            self.idct = nn.Identity()
        elif 'dct_torch' in cs:
            self.dct = DCT2x_torch()
            self.idct = IDCT2x_torch()
        else:
            self.dct = DCT2x()
            self.idct = IDCT2x()

        # 修改后的注意力模块
        self.norm1 = LayerNorm(dim, bias=True, mu_sigma=False)
        self.attn = ImprovedAttention(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            window_size=window_size
        )
        
    def filter_window_fast(self, x,threshold=0.7):
        B, C, H, W = x.shape
        center = x[:, :, H//2, W//2].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        norm_center = torch.norm(center, dim=1, keepdim=True)
        norm_x = torch.norm(x, dim=1, keepdim=True)
        similarity = (x * center).sum(dim=1, keepdim=True) / (norm_center * norm_x + 1e-6)
        mask = (similarity >= threshold).float()
        return x * mask

    def forward(self, x):
        x=self.filter_window_fast(x)
        x_dct = self.dct(x)
        x_attn = self.attn(self.norm1(x_dct))
        x_dct = x_dct + x_attn
        return self.idct(x_dct)
    
    


class Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        with open('dataset_info.yaml', 'r') as file:
            data = yaml.safe_load(file)
        data = data[dataset]

        self.out_features = data['num_classes']
        self.pca_num = data['pca_num']
        hsi_channel_num = data['hsi_channel_num']
        slar_channel_num = data['slar_channel_num']
        window_size = data["window_size"]

        self.FreqLCBlockHsiPca = EnhancedFreqLCBlock(
            dim=data['pca_num'],
            num_heads=1,  # 新增参数
            bias=False,  # 新增参数
            LayerNorm_type='WithBias',  # 新增参数
            window_size=data["window_size"],
            cs='dct'
        )
        self.FreqLCBlockSar = EnhancedFreqLCBlock(
            dim=data['slar_channel_num'],
            num_heads=1,  # 新增参数
            bias=False,  # 新增参数
            LayerNorm_type='WithBias',  # 新增参数
            window_size=data["window_size"],
            cs='dct'
        )

        self.linear_fusionPca = nn.Linear(in_features=(self.pca_num + slar_channel_num) * window_size * window_size,
                                          out_features=self.out_features)
        self.linear_fusion = nn.Linear(in_features=(hsi_channel_num + slar_channel_num) * window_size * window_size,
                                       out_features=self.out_features)

    def forward(self, hsi, sar):
        # print(f"input hsi shape: {hsi.shape}")
        # print(f"input sar shape: {sar.shape}")
        # 5d->4d
        hsi = hsi.squeeze(dim=1)
        hsi = self.FreqLCBlockHsiPca(hsi)
        sar = self.FreqLCBlockSar(sar)

        fusion_feat_m = torch.cat((hsi, sar), dim=1)
        B, _, _, _ = fusion_feat_m.size()
        fusion_feat = fusion_feat_m.reshape(B, -1)
        output_fusion = self.linear_fusionPca(fusion_feat)

        # print(f"output fusion_feat_m shape: {fusion_feat_m.shape}")
        # print(f"output output_fusion shape: {output_fusion.shape}")

        return fusion_feat_m, output_fusion
