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

# class TEFlowMGF(nn.Module):
#     def __init__(self, traj_c=16, flow_c=2, hidden=32):
#         super().__init__()
#         # 编码静态轨迹（空间）
#         self.traj_embed = nn.Conv2d(traj_c, traj_c, 3, padding=1)
#         # 编码光流（空间）
#         self.flow_spatial = nn.Conv2d(flow_c, hidden, 3, padding=1)
#         # 时间建模（关键）
#         self.flow_temporal_gamma = zero_module(
#             nn.Conv1d(hidden, traj_c, kernel_size=3, padding=1)
#         )
#         self.flow_temporal_beta = zero_module(
#             nn.Conv1d(hidden, traj_c, kernel_size=3, padding=1)
#         )
#         self.norm = nn.GroupNorm(4, traj_c)

#     def forward(self, traj, flow, control_scale):
#         """
#         traj: [B, C, H, W]
#         flow: [B, 2, T, H, W]
#         """
#         B, _, T, H, W = flow.shape
#         # ---- 1. 轨迹：静态空间编码 ----
#         traj_feat = self.traj_embed(traj)  # [B, C, H, W]
#         traj_feat = traj_feat.unsqueeze(2)  # [B, C, 1, H, W]
#         # ---- 2. 光流：空间 → 时间 ----
#         flow_ = rearrange(flow, "b c t h w -> (b t) c h w")
#         flow_feat = self.flow_spatial(flow_)  # [(B*T), hidden, H, W]
#         flow_feat = rearrange(
#             flow_feat, "(b t) c h w -> (b h w) c t", t=T
#         )
#         gamma = self.flow_temporal_gamma(flow_feat)  # [(B*H*W), C, T]
#         beta = self.flow_temporal_beta(flow_feat)
#         gamma = rearrange(
#             gamma, "(b h w) c t -> b c t h w", b=B, h=H, w=W
#         )
#         beta = rearrange(
#             beta, "(b h w) c t -> b c t h w", b=B, h=H, w=W
#         )
#         # ---- 3. 静态 → 动态（FiLM）----
#         traj_dynamic = traj_feat + control_scale * ( self.norm(traj_feat) * gamma + beta )
#         return traj_dynamic
class FlowConditionedTE(nn.Module):
    """
    将光流编码为条件信号，通过 FiLM 调制轨迹视频特征。
    轨迹视频已经是逐帧的 [B, C, T, H, W]，不需要 warp。
    光流提供全局视角运动的补充信息。
    """
    def __init__(self, traj_c=16, flow_c=2, hidden=32):
        super().__init__()
        # 光流空间编码
        self.flow_spatial = nn.Conv2d(flow_c, hidden, 3, padding=1)
        # 光流时间建模 → 生成 FiLM 参数
        self.flow_temporal_gamma = zero_module(
            nn.Conv1d(hidden, traj_c, kernel_size=3, padding=1)
        )
        self.flow_temporal_beta = zero_module(
            nn.Conv1d(hidden, traj_c, kernel_size=3, padding=1)
        )
        self.norm = nn.GroupNorm(4, traj_c)

    def forward(self, traj_video, flow, control_scale):
        """
        traj_video: [B, C, T, H, W]  — 已经是逐帧轨迹视频的 VAE latent
        flow:       [B, 2, T, H, W]  — 全局光流
        """
        B, C, T, H, W = traj_video.shape

        # 光流编码：空间 → 时间
        flow_ = rearrange(flow, "b c t h w -> (b t) c h w")
        flow_feat = self.flow_spatial(flow_)           # [(BT), hidden, H, W]
        flow_feat = rearrange(flow_feat, "(b t) c h w -> (b h w) c t", t=T)

        gamma = self.flow_temporal_gamma(flow_feat)     # [(BHW), C, T]
        beta  = self.flow_temporal_beta(flow_feat)

        gamma = rearrange(gamma, "(b h w) c t -> b c t h w", b=B, h=H, w=W)
        beta  = rearrange(beta,  "(b h w) c t -> b c t h w", b=B, h=H, w=W)

        # FiLM 调制：轨迹 + 光流条件
        out = traj_video + control_scale * (self.norm(traj_video) * gamma + beta)
        return out


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
        self.flow_modulator = FlowConditionedTE(traj_c=cin)
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
        zero_module(self.flow_modulator.flow_temporal_gamma)
        zero_module(self.flow_modulator.flow_temporal_beta)

    def forward(self, traj_latent, flow, warmup_scale = 1.0):
        """
        x: torch.Tensor: shape [B C T H W]
        """
        x = self.flow_modulator(traj_latent, flow.detach(), warmup_scale)
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
                # print(torch.sum(x))
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
