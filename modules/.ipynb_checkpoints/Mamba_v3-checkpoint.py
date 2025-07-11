import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat



NEG_INF = -1000000




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) #(b,k*d,l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        #需要反转回同样的方向
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)#对不是核心维度进行全连接
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()  #(B,C,H,W)
        x = self.act(self.conv2d(x))  
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class MS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1., 
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model   #可以理解为通道数量是多少
        self.d_state = d_state
        self.d_conv = d_conv   #卷积核的大小
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  #这里有一个expand=2
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) #对输入的通道数进行投影一下
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        #保持原尺度大小的DWConv
        self.DWconv1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            padding=(3 - 1) // 2,
            **factory_kwargs,
        )
        #缩小尺度大小的DWConv
        self.DWconv2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=5,
            padding=(5-1)//2,
            stride=2,
            **factory_kwargs,
        )
        
        #使用转置卷积来上采样
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=self.d_inner,        # 输入通道数
            out_channels=self.d_inner,       # 输出通道数
            kernel_size=5,        # 卷积核大小
            stride=2,             # 步幅
            padding=(5-1)//2,            # 填充
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank),取前3个
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=False)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=False)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=False):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core_13(self, x: torch.Tensor):
        B, C, H, W = x.shape
        #一个方向，使用一个DWConv
        x_pri = self.DWconv1(x)
        x_1 = x_pri.view(B, -1, H * W)  #序列1
        #剩下三个方向要缩小
        x_rem = self.DWconv2(x)
        B1, C1, H1, W1 = x_rem.shape
        x_2 = torch.transpose(x_rem,dim0=2, dim1=3).contiguous().view(B, -1, H1*W1).unsqueeze(1)
        x_3 = torch.flip(x_rem.view(B, -1, H1 * W1),dims=[-1]).unsqueeze(1)
        x_4 = torch.flip(x_2,dims=[-1])
        #首先生成x_pri的辅助数据
        x_pri_tbc = torch.einsum("b d l, c d -> b c l", x_1, self.x_proj_weight[0])
        # x_pri_tbc = self.x_proj[0](x_1.permute(0,2,1).contiguous()).permute(0,2,1)
        dts_pri, Bs_pri, Cs_pri = torch.split(x_pri_tbc, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts_pri = torch.einsum("b c l, d c -> b d l", dts_pri, self.dt_projs_weight[0])
        
        A_pri = -torch.exp(self.A_logs[0].float()).view(-1, self.d_state)
        dt_projs_bias_pri = self.dt_projs_bias[0].float().view(-1) 
        out_y_pri = self.selective_scan(
            x_1, dts_pri,
            A_pri, Bs_pri, Cs_pri, self.Ds[0], z=None,
            delta_bias=dt_projs_bias_pri,
            delta_softplus=True,
            return_last_state=False,
        )
        assert out_y_pri.dtype == torch.float
        #由于第一种方向就是原始展开的方向所以就不需要反转回原来的方向了
        # out_y_1 = 
        #先将三种方向堆叠一下
        x_234=torch.cat([x_2, x_3, x_4], dim=1)
        #生成x_rem的辅助数据
        x_rem_tbc = torch.einsum("b k d l,k c d -> b k c l", x_234, self.x_proj_weight[1:])
        dts_rem, Bs_rem, Cs_rem = torch.split(x_rem_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_rem = torch.einsum("b k c l,k d c -> b k d l", dts_rem, self.dt_projs_weight[1:])
        # u: r(B D L)
        # delta: r(B D L)
        # A: c(D N) or r(D N)
        # B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        # C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        # D: r(D)
        # z: r(B D L)
        # delta_bias: r(D), fp32
        x_234 = x_234.reshape(x_234.shape[0],-1,x_234.shape[3])
        A_rem = -torch.exp(self.A_logs[1:].float()).view(-1, self.d_state).contiguous()
        D_rem = self.Ds[1:].reshape(-1)
        dts_rem = dts_rem.reshape(dts_rem.shape[0],-1,dts_rem.shape[3])
        dt_projs_bias_rem = self.dt_projs_bias[1:].float().view(-1) 
        out_y_rem = self.selective_scan(
            x_234, dts_rem,
            A_rem, Bs_rem, Cs_rem, D_rem, z=None,
            delta_bias=dt_projs_bias_rem,
            delta_softplus=True,
            return_last_state=False,
        )
        assert out_y_rem.dtype == torch.float
        #需要反转回同样的方向
        out_y_rem = out_y_rem.reshape(out_y_rem.shape[0],3,-1,out_y_rem.shape[2])
        out_y_2 = torch.transpose(out_y_rem[:,0].reshape(B1,C1,W1,H1),dim0=2,dim1=3)
        out_y_3 = torch.flip(out_y_rem[:,1],dims=[-1]).reshape(B1,C1,H1,W1)
        out_y_4 = torch.transpose(torch.flip(out_y_rem[:,2],dims=[-1]).reshape(B1,C1,W1,H1),dim0=2,dim1=3)
        out_y_2 = self.conv_transpose(out_y_2).reshape(B,C,-1)  #公用一个转置卷积
        out_y_3 = self.conv_transpose(out_y_3).reshape(B,C,-1)
        out_y_4 = self.conv_transpose(out_y_4).reshape(B,C,-1)
        return out_y_pri,out_y_2,out_y_3,out_y_4

    def forward_core_22(self, x: torch.Tensor):
        B, C, H, W = x.shape
        #一个方向，使用一个DWConv
        x_pri = self.DWconv1(x)
        x_1 = x_pri.view(B, -1, H * W).unsqueeze(1)  #序列1
        x_2 = torch.transpose(x_pri,dim0=2, dim1=3).contiguous().view(B,-1,H*W).unsqueeze(1)#序列2
        x_12 = torch.cat([x_1,x_2],dim=1)
        #剩下2个方向要缩小
        x_rem = self.DWconv2(x)
        B1, C1, H1, W1 = x_rem.shape
        # x_2 = torch.transpose(x_rem,dim0=2, dim1=3).contiguous().view(B, -1, H1*W1).unsqueeze(1)
        x_3 = torch.flip(x_rem.view(B, -1, H1 * W1),dims=[-1]).unsqueeze(1)
        x_4 = torch.flip(torch.transpose(x_rem,dim0=2, dim1=3).reshape(B,-1,H1*W1),dims=[-1]).unsqueeze(1)
        #首先生成x_pri的辅助数据
        x_pri_tbc = torch.einsum("b k d l,k c d ->b k c l", x_12, self.x_proj_weight[:2])
        # x_pri_tbc = self.x_proj[0](x_1.permute(0,2,1).contiguous()).permute(0,2,1)
        dts_pri, Bs_pri, Cs_pri = torch.split(x_pri_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_pri = torch.einsum("b k c l, k d c -> b k d l", dts_pri, self.dt_projs_weight[:2])
        
        A_pri = -torch.exp(self.A_logs[:2].float()).view(-1, self.d_state)
        dt_projs_bias_pri = self.dt_projs_bias[:2].float().view(-1) 
        x_12 = x_12.view(B, -1, H*W)
        dts_pri = dts_pri.reshape(B,-1,H*W)
        out_y_pri = self.selective_scan(
            x_12, dts_pri,
            A_pri, Bs_pri, Cs_pri, self.Ds[:2].reshape(-1), z=None,
            delta_bias=dt_projs_bias_pri,
            delta_softplus=True,
            return_last_state=False,
        )
        assert out_y_pri.dtype == torch.float
        #由于第一种方向就是原始展开的方向所以就不需要反转回原来的方向了
        # out_y_1 = 
        #先将三种方向堆叠一下
        x_34 = torch.cat([x_3, x_4], dim=1)
        #生成x_rem的辅助数据
        x_rem_tbc = torch.einsum("b k d l,k c d -> b k c l", x_34, self.x_proj_weight[2:])
        dts_rem, Bs_rem, Cs_rem = torch.split(x_rem_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_rem = torch.einsum("b k c l,k d c -> b k d l", dts_rem, self.dt_projs_weight[2:])
        # u: r(B D L)
        # delta: r(B D L)
        # A: c(D N) or r(D N)
        # B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        # C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        # D: r(D)
        # z: r(B D L)
        # delta_bias: r(D), fp32
        x_34 = x_34.reshape(x_34.shape[0],-1,x_34.shape[3])
        A_rem = -torch.exp(self.A_logs[2:].float()).view(-1, self.d_state).contiguous()
        D_rem = self.Ds[2:].reshape(-1)
        dts_rem = dts_rem.reshape(dts_rem.shape[0],-1,dts_rem.shape[3])
        dt_projs_bias_rem = self.dt_projs_bias[2:].float().view(-1) 
        out_y_rem = self.selective_scan(
            x_34, dts_rem,
            A_rem, Bs_rem, Cs_rem, D_rem, z=None,
            delta_bias=dt_projs_bias_rem,
            delta_softplus=True,
            return_last_state=False,
        )
        assert out_y_rem.dtype == torch.float
        #需要反转回同样的方向
        out_y_pri = out_y_pri.reshape(B,2,-1,H * W)
        out_y_rem = out_y_rem.reshape(B,2,-1,H1 * W1)
        out_y_1 = out_y_pri[:,0]
        out_y_2 = torch.transpose(out_y_pri[:,1].reshape(B,C,W,H),dim0=2,dim1=3).reshape(B,C,-1)
        out_y_3 = torch.flip(out_y_rem[:,0],dims=[-1]).reshape(B1,C1,H1,W1)
        out_y_4 = torch.transpose(torch.flip(out_y_rem[:,1],dims=[-1]).reshape(B1,C1,W1,H1),dim0=2,dim1=3)
        #公用一个转置卷积
        out_y_3 = self.conv_transpose(out_y_3).reshape(B,C,-1)
        out_y_4 = self.conv_transpose(out_y_4).reshape(B,C,-1)
        return out_y_1,out_y_2,out_y_3,out_y_4

    def forward_core_31(self, x: torch.Tensor):
        B, C, H, W = x.shape
        #一个方向，使用一个DWConv
        x_pri = self.DWconv1(x)
        x_1 = x_pri.view(B, -1, H * W).unsqueeze(1)  #序列1
        x_2 = torch.transpose(x_pri, dim0=2, dim1=3).contiguous().view(B,-1,H*W).unsqueeze(1)#序列2
        x_3 = torch.flip(x_1,dims = [-1]) #序列3
        x_123 = torch.cat([x_1, x_2, x_3],dim=1)
        #剩下2个方向要缩小
        x_rem = self.DWconv2(x)
        B1, C1, H1, W1 = x_rem.shape
        x_4 = torch.flip(torch.transpose(x_rem,dim0=2, dim1=3).reshape(B, -1, H1 * W1),dims=[-1]).unsqueeze(1)
        #首先生成x_pri的辅助数据
        x_pri_tbc = torch.einsum("b k d l,k c d ->b k c l", x_123, self.x_proj_weight[:3])
        # x_pri_tbc = self.x_proj[0](x_1.permute(0,2,1).contiguous()).permute(0,2,1)
        dts_pri, Bs_pri, Cs_pri = torch.split(x_pri_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_pri = torch.einsum("b k c l, k d c -> b k d l", dts_pri, self.dt_projs_weight[:3])
        
        A_pri = -torch.exp(self.A_logs[:3].float()).view(-1, self.d_state)
        dt_projs_bias_pri = self.dt_projs_bias[:3].float().view(-1) 
        x_123 = x_123.view(B, -1, H * W)
        dts_pri = dts_pri.reshape(B,-1, H * W)
        out_y_pri = self.selective_scan(
            x_123, dts_pri,
            A_pri, Bs_pri, Cs_pri, self.Ds[:3].reshape(-1), z=None,
            delta_bias=dt_projs_bias_pri,
            delta_softplus=True,
            return_last_state=False,
        )
        assert out_y_pri.dtype == torch.float

        #生成x_rem的辅助数据
        x_rem_tbc = torch.einsum("b k d l,k c d -> b k c l", x_4, self.x_proj_weight[3].unsqueeze(0))
        dts_rem, Bs_rem, Cs_rem = torch.split(x_rem_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_rem = torch.einsum("b k c l,k d c -> b k d l", dts_rem, self.dt_projs_weight[3].unsqueeze(0))
        # u: r(B D L)
        # delta: r(B D L)
        # A: c(D N) or r(D N)
        # B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        # C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        # D: r(D)
        # z: r(B D L)
        # delta_bias: r(D), fp32
        x_4 = x_4.reshape(x_4.shape[0], -1, x_4.shape[3])
        A_rem = -torch.exp(self.A_logs[3].float()).view(-1, self.d_state).contiguous()
        D_rem = self.Ds[3].reshape(-1)
        dts_rem = dts_rem.reshape(dts_rem.shape[0],-1,dts_rem.shape[3])
        dt_projs_bias_rem = self.dt_projs_bias[3].float().view(-1) 
        out_y_rem = self.selective_scan(
            x_4, dts_rem,
            A_rem, Bs_rem, Cs_rem, D_rem, z=None,
            delta_bias=dt_projs_bias_rem,
            delta_softplus=True,
            return_last_state=False,
        )
        assert out_y_rem.dtype == torch.float
        #需要反转回同样的方向
        out_y_pri = out_y_pri.reshape(B,3,-1,H * W)
        out_y_rem = out_y_rem.reshape(B,1,-1,H1 * W1)
        out_y_1 = out_y_pri[:,0]
        out_y_2 = torch.transpose(out_y_pri[:,1].reshape(B,C,W,H),dim0=2,dim1=3).reshape(B,C,-1)
        out_y_3 = torch.flip(out_y_pri[:,2],dims=[-1])
        out_y_4 = torch.transpose(torch.flip(out_y_rem[:,0],dims=[-1]).reshape(B1, C1, W1, H1),dim0=2,dim1=3)

        out_y_4 = self.conv_transpose(out_y_4).reshape(B,C,-1)
        return out_y_1,out_y_2,out_y_3,out_y_4
    
    def forward_core_04(self, x: torch.Tensor):
            B, C, H, W = x.shape
            #剩下2个方向要缩小
            x_rem = self.DWconv2(x)
            B1, C1, H1, W1 = x_rem.shape
            x_1 = x_rem.view(B, -1, H1 * W1).unsqueeze(1)  #序列1
            x_2 = torch.transpose(x_rem,dim0=2, dim1=3).contiguous().view(B,-1,H1*W1).unsqueeze(1)#序列2
            x_3 = torch.flip(x_rem.view(B, -1, H1 * W1),dims=[-1]).unsqueeze(1)
            x_4 = torch.flip(torch.transpose(x_rem,dim0=2, dim1=3).reshape(B,-1,H1 * W1),dims=[-1]).unsqueeze(1)
            
            x_1234 = torch.cat([x_1,x_2,x_3, x_4], dim=1)
            #生成x_rem的辅助数据
            x_rem_tbc = torch.einsum("b k d l,k c d -> b k c l", x_1234, self.x_proj_weight)
            dts_rem, Bs_rem, Cs_rem = torch.split(x_rem_tbc, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts_rem = torch.einsum("b k c l,k d c -> b k d l", dts_rem, self.dt_projs_weight)
            # u: r(B D L)
            # delta: r(B D L)
            # A: c(D N) or r(D N)
            # B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
            # C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
            # D: r(D)
            # z: r(B D L)
            # delta_bias: r(D), fp32
            x_1234 = x_1234.reshape(x_1234.shape[0],-1,x_1234.shape[3])
            A_rem = -torch.exp(self.A_logs.float()).view(-1, self.d_state).contiguous()
            D_rem = self.Ds.reshape(-1)
            dts_rem = dts_rem.reshape(dts_rem.shape[0],-1,dts_rem.shape[3])
            dt_projs_bias_rem = self.dt_projs_bias.float().view(-1) 
            out_y_rem = self.selective_scan(
                x_1234, dts_rem,
                A_rem, Bs_rem, Cs_rem, D_rem, z=None,
                delta_bias=dt_projs_bias_rem,
                delta_softplus=True,
                return_last_state=False,
            )
            assert out_y_rem.dtype == torch.float
            #需要反转回同样的方向
            out_y_rem = out_y_rem.reshape(B1, 4, -1, H1 * W1)
            out_y_1 = out_y_rem[: , 0].reshape(B1, C1, H1, W1)
            out_y_2 = torch.transpose(out_y_rem[: , 1].reshape(B1, C1, W1, H1),dim0=2,dim1=3)
            out_y_3 = torch.flip(out_y_rem[:,2],dims=[-1]).reshape(B1,C1,H1,W1)
            out_y_4 = torch.transpose(torch.flip(out_y_rem[:,3],dims=[-1]).reshape(B1,C1,W1,H1),dim0=2,dim1=3)
            #公用一个转置卷积
            out_y_1 = self.conv_transpose(out_y_1).reshape(B,C,-1)
            out_y_2 = self.conv_transpose(out_y_2).reshape(B,C,-1)
            out_y_3 = self.conv_transpose(out_y_3).reshape(B,C,-1)
            out_y_4 = self.conv_transpose(out_y_4).reshape(B,C,-1)
            return out_y_1,out_y_2,out_y_3,out_y_4


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)#对不是核心维度进行全连接 有对维度乘2的操作 后期可以考虑去掉
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()  #(B,C,H,W)
        x = self.act(self.conv2d(x))  
        #在这里执行多尺度操作
        y1, y2, y3, y4 = self.forward_core_22(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class Fuse_SS2D(nn.Module):
    def __init__(
            self,
            d_model1,   #输入x的通道维度
            d_model2, #输入y的通道维度 
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model1 = d_model1
        self.d_model2 = d_model2
        self.d_state = d_state
        self.expand = expand
        self.d_inner1 = int(self.expand * self.d_model1)
        self.d_inner2 = int(self.expand * self.d_model2)
        #输出的结果只与y相关，所以这里需要输入x的信息，输出y匹配的维度
        self.dt_rank = math.ceil(self.d_model1 / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  #(K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner2, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner2, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner2)
        # self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x, y):  #x和y的形状一样 y和out_y没有关系的 输出结果与y相关
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        #增加输入的ys
        y_hwwh = torch.stack([y.view(B, -1, L), torch.transpose(y, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        ys = torch.cat([y_hwwh, torch.flip(y_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        ys = ys.float().view(B, -1, L)
        #最后输入的是y，输出的也与y相关
        out_y = self.selective_scan(
            ys, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x, y):  #第一个是HSI第二个是Lidar/SAR
        B, C, H, W = x.shape
        
        ya1, ya2, ya3, ya4 = self.forward_core(x ,y)
        
        ya = ya1 + ya2 + ya3 + ya4
        ya = torch.transpose(ya, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        ya = self.out_norm(ya)
        return ya

class FSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim1: int = 0,
            hidden_dim2: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 0.5,
            d_conv=3,
            bias=False,
            conv_bias=True,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim1)
        self.ln_2 = norm_layer(hidden_dim2)
        self.expand = expand
        self.d_inner1 = int(self.expand * hidden_dim1)
        self.d_inner2 = int(self.expand * hidden_dim2)
        
        self.in_proj1 = nn.Linear(hidden_dim1, self.d_inner1 * 2, bias=bias)
        self.in_proj2 = nn.Linear(hidden_dim2, self.d_inner2 * 2, bias=bias)
        
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner1,
            out_channels=self.d_inner1,
            groups=self.d_inner1,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner2,
            out_channels=self.d_inner2,
            groups=self.d_inner2,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()
        
        self.attention1 = Fuse_SS2D(d_model1=hidden_dim1, d_model2=hidden_dim2, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)#代码中的SS2D完成的内容更多
        self.attention2 = Fuse_SS2D(d_model1=hidden_dim2, d_model2=hidden_dim1, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.out_proj1 = nn.Linear(self.d_inner1, hidden_dim1, bias=bias)
        self.out_proj2 = nn.Linear(self.d_inner2, hidden_dim2, bias=bias)
        self.drop_path = DropPath(drop_path)
        

    def forward(self, x, y):#先输入HSI再输入SAR/Lidar
        dim_num=len(x.size())
        B,C,N,H,W = 0,0,0,0,0
        if(dim_num==5):
            B,C,N,H,W=x.size()
            x=x.reshape(B,C*N,H,W)
        else:
            B,C,H,W=x.size()
        #对通道维度进行layerNorm
        x = x.reshape(B,H,W,-1)
        y = y.reshape(B,H,W,-1)
        
        x_ = self.ln_1(x)
        y_ = self.ln_2(y)
        
        x_12 = self.in_proj1(x_)
        x_1, x_2 = x_12.chunk(2, dim=-1)
        
        y_12 = self.in_proj2(y_)
        y_1, y_2 = y_12.chunk(2, dim=-1)
        #对x_1和x_2进行维度变化
        x_1=x_1.reshape(B,-1,H,W)
        y_1=y_1.reshape(B,-1,H,W)
        
        x_1=self.act(self.conv2d1(x_1))
        y_1=self.act(self.conv2d2(y_1))
        
        #先投影分成2个部分
        y_out = self.attention1(x_1 , y_1)
        x_out = self.attention2(y_1 , x_1)
        
        x_out = x_out * F.silu(x_2)
        y_out = y_out * F.silu(y_2)
        
        out_x=self.out_proj1(x_out)
        out_y=self.out_proj2(y_out)
        
        x = x + out_x
        y = y + out_y
        #再将维度变换
        x = x.reshape(B,-1,H,W)
        y = y.reshape(B,-1,H,W)
        if(dim_num==5):
            x=x.reshape(B,C,N,H,W)
        return x,y

class MSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 0.5,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input):
        # h [B,C,N,H,W]
        # x [B,C,H,W]
        cate='x'
        B,C,N,H,W=0,0,0,0,0
        #得到input的维度
        if(len(input.size())==5):
            #对输入数据维度进行变换
            B,C,N,H,W=input.size()
            input=input.reshape(B,C*N,H,W)
            cate='h'
        else:
            B,C,H,W=input.size()
        #对通道维度进行layerNorm,并且通道都是在最后一个维度
        input=input.reshape(B,H,W,-1)
        x = self.ln_1(input)
        x = self.self_attention(x)  #包含Linear到Linear
        x = input + x
        x = x.permute(0, 3, 1, 2).contiguous()
        if(cate=='h'):
            x=x.reshape(B,C,N,H,W)
        return x


#先完成一个普通的Mamba Block
class MambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)#代码中的SS2D完成的内容更多
        self.drop_path = DropPath(drop_path)

    def forward(self, input):
        # h [B,C,N,H,W]
        # x [B,C,H,W]
        cate='x'
        B,C,N,H,W=0,0,0,0,0
        #得到input的维度
        if(len(input.size())==5):
            #对输入数据维度进行变换
            B,C,N,H,W=input.size()
            input=input.reshape(B,C*N,H,W)
            cate='h'
        else:
            B,C,H,W=input.size()
        #对通道维度进行layerNorm,并且通道都是在最后一个维度
        input=input.reshape(B,H,W,-1)
        x = self.ln_1(input)
        x = self.self_attention(x)  #包含Linear到Linear
        x = input + x
        x = x.permute(0, 3, 1, 2).contiguous()
        if(cate=='h'):
            x=x.reshape(B,C,N,H,W)
        return x


class Spec_SS1D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.proj = nn.Linear(self.d_inner, self.d_inner, bias=bias)
        self.act = nn.SiLU()

        self.x_proj = (  #通道维度只有两个方向去掉一个
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, D = x.shape
        L = C
        K = 2
        xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        #这里需要修改，加上反转操作
        y2 = torch.flip(out_y[:,1], dims=[-1])
        return out_y[:,0], y2

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.act(self.proj(x))
        y1, y2 = self.forward_core(x)  #核心
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = torch.transpose(y, dim0=1, dim1=2).contiguous()
        #y=y.view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    

#完成spectralMamba
class SpecMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 0.5,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = Spec_SS1D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input):#只针对5维的高光谱图像
        # h [B,C,N,H,W]
        B,C,N,H,W=input.size()
        input=input.reshape(B,C*N,H*W) # (因为Mamba块那里说通道是最后一维，dim也示意最后一维，此处顺序会报错，所以我改了一下）
        #input = input.reshape(B, H * W, C * N)
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = x + input
        
        x = x.reshape(B,C,N,H,W)
        return x


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)