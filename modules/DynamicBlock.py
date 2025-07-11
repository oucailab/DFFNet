import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=bias
        )
        self.bn2 = nn.BatchNorm2d(out_channels) # 对逐点卷积的输出进行BN
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class FFN(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.dim = dim # 通道数

        # 分支2: 空间域卷积
        self.spatial_conv1 = DepthwiseSeparableConv(dim, dim, bias=bias)
        self.spatial_conv2 = DepthwiseSeparableConv(dim, dim, bias=bias)

        # 分支3: 频率域卷积 (作用于实部和虚部)
        self.freq_conv_real = DepthwiseSeparableConv(dim, dim, bias=bias) # 处理实部
        self.freq_conv_imag = DepthwiseSeparableConv(dim, dim, bias=bias) # 处理虚部

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim, "输入通道数与FFN定义的dim不匹配"

        # 分支1: 恒等分支
        identity_branch = x

        # 分支2: 空间域处理
        spatial_branch = self.spatial_conv2(self.spatial_conv1(x))

        # 分支3: 频率域处理
        x_freq_complex = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho') #(B, C, H, W_f = W//2 + 1)

        # 分离实部和虚部
        x_freq_real = x_freq_complex.real # (B, C, H, W_f)
        x_freq_imag = x_freq_complex.imag # (B, C, H, W_f)

        # 用 DepthwiseSeparableConv 分别处理实部和虚部
        processed_real = self.freq_conv_real(x_freq_real)
        processed_imag = self.freq_conv_imag(x_freq_imag)

        recombined_freq_complex = torch.complex(processed_real, processed_imag)

        freq_branch = torch.fft.irfft2(recombined_freq_complex, s=(H, W), dim=(-2, -1), norm='ortho')

        out = identity_branch + spatial_branch + freq_branch
        return out

class DynamicFilterBlock(nn.Module):
    def __init__(self, dim, num_filter_bases=4, mlp_ratio=2, bias=False):
        super().__init__()
        self.dim = dim
        self.num_filter_bases = num_filter_bases

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_filter_generator = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, num_filter_bases, bias=bias)
        )

        self.ffn = FFN(dim, bias=bias)
        self.initialized_bases = False

    def _initialize_filter_bases(self,device):
        init_val = torch.randn(self.num_filter_bases, self.dim, 1, 1, 2, device=device) * 0.02
        self.filter_bases_F = nn.Parameter(init_val)  # 可学习的基滤波器
        self.initialized_bases = True


    def forward(self, x):
        # 1. 输入预处理
        B, C, H, W = x.shape
        assert C == self.dim, f"输入通道数 {C} 与 DFB 维度 {self.dim} 不匹配"

        H_freq = H
        W_freq_rfft = W // 2 + 1

        x_freq_complex = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')

        if not self.initialized_bases:
            self._initialize_filter_bases(x.device)

        # 2. 动态滤波器生成
        context_vector = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)

        filter_component_weights = self.mlp_filter_generator(context_vector)
        filter_component_weights = F.softmax(filter_component_weights, dim=-1)

        filter_bases_expanded = self.filter_bases_F.expand(-1, -1, H_freq, W_freq_rfft, -1)

        K_xin_real_imag = torch.einsum('bn,nchwr->bchwr', filter_component_weights, filter_bases_expanded)
        K_xin_complex = torch.view_as_complex(K_xin_real_imag.contiguous())

        # 应用动态滤波器
        dynamically_filtered_freq_complex = x_freq_complex * K_xin_complex

        x_out_spatial_filtered = torch.fft.irfft2(dynamically_filtered_freq_complex, s=(H, W), dim=(-2, -1), norm='ortho') + x

        output = self.ffn(x_out_spatial_filtered)

        return output