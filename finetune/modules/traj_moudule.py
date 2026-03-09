# Copyright 2024 Alibaba Group.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator
from functools import reduce as reduce_

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        self.bn1 = nn.GroupNorm(8, out_c)
        self.bn2 = nn.GroupNorm(8, out_c)
        if sk == False:
            # self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps) # edit by zhouxiawang
            self.skep = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.bn1(x)
        h = self.act(h)
        h = self.block1(h)
        h = self.bn2(h)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class VAESpatialEmulator(nn.Module):
    def __init__(self, kernel_size=(8, 8)):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        x: torch.Tensor: shape [B C T H W]
        """
        Hp, Wp = self.kernel_size
        H, W = x.shape[-2], x.shape[-1]
        valid_h = H - H % Hp
        valid_w = W - W % Wp
        x = x[..., :valid_h, :valid_w]
        x = rearrange(
            x,
            "B C T (Nh Hp) (Nw Wp)  -> B (Hp Wp C) T Nh Nw",
            Hp=Hp,
            Wp=Wp,
        )
        return x


class VAETemporalEmulator(nn.Module):
    def __init__(self, micro_frame_size, kernel_size=4):
        super().__init__()
        self.micro_frame_size = micro_frame_size
        self.kernel_size = kernel_size

    def forward(self, x_z):
        """
        x_z: torch.Tensor: shape [B C T H W]
        """

        z_list = []
        for i in range(0, x_z.shape[2], self.micro_frame_size):
            x_z_bs = x_z[:, :, i : i + self.micro_frame_size]
            z_list.append(x_z_bs[:, :, 0:1])
            x_z_bs = x_z_bs[:, :, 1:]
            t_valid = x_z_bs.shape[2] - x_z_bs.shape[2] % self.kernel_size
            x_z_bs = x_z_bs[:, :, :t_valid]
            x_z_bs = reduce(x_z_bs, "B C (T n) H W -> B C T H W", n=self.kernel_size, reduction="mean")
            z_list.append(x_z_bs)
        z = torch.cat(z_list, dim=2)
        return z


def ensure_tuple(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    return xs


class Patchify3D(nn.Module):
    def __init__(self, patch_size=2):
        super().__init__()

        patch_size = ensure_tuple(patch_size, 3)
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: torch.Tensor: shape [B C T H W]
        out: torch.Tensor: shape [B (C Tp Hp Wp) Nt Nh Nw]
        """
        Tp, Hp, Wp = self.patch_size

        x = rearrange(
            x,
            "B C (Nt Tp) (Nh Hp) (Nw Wp)  -> B (C Tp Hp Wp) Nt Nh Nw",
            Tp=Tp,
            Hp=Hp,
            Wp=Wp,
        )

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTrajExtractor(nn.Module):
    """
    简化版轨迹提取器,专为固定尺寸光流设计
    输入: flow [B, 2, 13, 60, 90]
    输出: 多尺度特征 List[[BT, C, H, W]] 用于 MGF
    """
    def __init__(
        self,
        flow_c=2,
        num_frames=13,
        spatial_size=(60, 90),
        patch_size=2,
        patch_size_t=1,
        channels=[128, 128, 128],  # 输出通道,对应不同 Transformer 层
        nums_rb=2,
        use_temporal_attn=False,  # 可选:是否使用时间注意力
    ):
        super().__init__()
        
        self.channels = channels  # 保存为实例变量
        self.nums_rb = nums_rb

        # 1. 时空编码器:将光流编码为初始特征
        self.flow_encoder = nn.Sequential(
            # 3D 卷积:同时处理时空
            nn.Conv3d(flow_c, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        
        # 2. 可选:时间注意力(增强时序建模)
        self.use_temporal_attn = use_temporal_attn
        if use_temporal_attn:
            self.temporal_attn = TemporalSelfAttention(128, num_heads=4)
        
        # 3. Patchify:将 3D 特征转为 2D 特征序列
        self.patch_size = (patch_size_t, patch_size, patch_size)
        self.patchify = Patchify3D(self.patch_size)
        
        # 4. 2D ResNet blocks:提取多尺度空间特征
        cin = 128 * patch_size_t * patch_size * patch_size
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        
        self.body = nn.ModuleList()
        for i in range(len(channels)):
            for j in range(nums_rb):
                if i != 0 and j == 0:
                    self.body.append(
                        ResnetBlock(channels[i-1], channels[i], down=False, ksize=3, sk=True, use_conv=False)
                    )
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=3, sk=True, use_conv=False)
                    )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, flow, warmup_scale=1.0):
        """
        flow: [B, 2, T, H, W] - 光流,T=13, H=60, W=90
        warmup_scale: float - 控制强度(训练初期较小)
        
        返回: List[[BT, C, H', W']] - 多尺度特征,用于 MGF
        """
        B, _, T, H, W = flow.shape
        
        # 1. 时空编码
        x = self.flow_encoder(flow)  # [B, 128, T, H, W]
        
        # 2. 可选:时间注意力
        if self.use_temporal_attn:
            x = self.temporal_attn(x)
        
        # 3. 应用 warmup_scale(渐进式注入)
        x = x * warmup_scale
        
        # 4. Patchify:3D → 2D
        x = self.patchify(x)  # [B, C', T', H', W']
        x = rearrange(x, "B C T H W -> (B T) C H W")
        
        # 5. 2D ResNet:提取多尺度特征
        features = []
        x = self.conv_in(x)
        
        for i in range(len(self.channels)):  # 假设 nums_rb=2
            for j in range(self.nums_rb ):
                idx = i * self.nums_rb  + j
                x = self.body[idx](x)
            features.append(x)  # 每个尺度保存一次
        
        return features

class TrajExtractor(nn.Module):
    def __init__(
        self,
        vae_downsize=(4, 8, 8),
        patch_size=2,
        patch_size_t=1,# Tora = 2
        channels=[320, 640, 1280], # Tora = [320, 640, 1280, 1280]
        nums_rb=2, # Tora = 3
        cin=16,
        ksize=3,
        sk=False,
        use_conv=True,
    ):
        super(TrajExtractor, self).__init__()
        self.vae_downsize = vae_downsize
        # self.vae_spatial_emulator = VAESpatialEmulator(kernel_size=vae_downsize[-2:])
        self.patch_size = (patch_size_t, patch_size, patch_size)
        self.downsize_patchify = Patchify3D(self.patch_size)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(
                            channels[i - 1],
                            channels[i],
                            down=False,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                        )
                    )
                else:
                    self.body.append(
                        ResnetBlock(
                            channels[i],
                            channels[i],
                            down=False,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                        )
                    )
        self.body = nn.ModuleList(self.body)
        cin_ = cin * reduce_(operator.mul, self.patch_size)
        self.conv_in = nn.Conv2d(cin_, channels[0], 3, 1, 1)

        # Initialize weights
        def conv_init(module):
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(conv_init)

    def forward(self, flow, warmup_scale = 1.0):
        """
        x: torch.Tensor: shape [B C T H W]
        """
        B, C, T, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(
                x,
                (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]),
            )
        x = self.downsize_patchify(x)
        x = rearrange(x, "B C T H W -> (B T) C H W")

        # extract features
        features = []

        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features



class FloatGroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.to(self.bias.dtype)).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MGF(nn.Module):
    def __init__(self, flow_in_channel=128, out_channels=1152):
        super().__init__()
        self.out_channels = out_channels
        self.flow_gamma_spatial = nn.Conv2d(flow_in_channel, self.out_channels // 4, 3, padding=1)
        self.flow_gamma_temporal = zero_module(
            nn.Conv1d(
                self.out_channels // 4,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            )
        )
        self.flow_beta_spatial = nn.Conv2d(flow_in_channel, self.out_channels // 4, 3, padding=1)
        self.flow_beta_temporal = zero_module(
            nn.Conv1d(
                self.out_channels // 4,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            )
        )
        self.flow_cond_norm = FloatGroupNorm(32, self.out_channels)

    def forward(self, h, flow, T):
        if flow is not None:
            gamma_flow = self.flow_gamma_spatial(flow)
            beta_flow = self.flow_beta_spatial(flow)
            _, _, hh, wh = beta_flow.shape
            gamma_flow = rearrange(gamma_flow, "(b f) c h w -> (b h w) c f", f=T)
            beta_flow = rearrange(beta_flow, "(b f) c h w -> (b h w) c f", f=T)
            gamma_flow = self.flow_gamma_temporal(gamma_flow)
            beta_flow = self.flow_beta_temporal(beta_flow)
            gamma_flow = rearrange(gamma_flow, "(b h w) c f -> (b f) c h w", h=hh, w=wh)
            beta_flow = rearrange(beta_flow, "(b h w) c f -> (b f) c h w", h=hh, w=wh)
            h = h + self.flow_cond_norm(h) * gamma_flow + beta_flow
        return h
