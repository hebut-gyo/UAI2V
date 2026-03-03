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


import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowWarpModulator(nn.Module):
    """
    基于光流 Warp 的轨迹调制器
    
    核心思想：使用光流对静态轨迹进行空间变换
    """
    def __init__(self, traj_c=16, flow_c=2, hidden=64):
        super().__init__()
        
        # 光流预处理网络
        self.flow_refine = nn.Sequential(
            nn.Conv3d(flow_c, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(),
            nn.Conv3d(hidden, flow_c, 3, padding=1)
        )
        
        # 特征增强网络
        self.feature_enhance = nn.Sequential(
            nn.Conv3d(traj_c, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(),
            nn.Conv3d(hidden, traj_c, 3, padding=1)
        )
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Conv3d(traj_c * 2, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(),
            nn.Conv3d(hidden, traj_c, 3, padding=1)
        )
    
    def forward(self, traj_latent, flow, control_scale=1.0):
        """
        Args:
            traj_latent: [B, 16, T, H, W] 静态轨迹
            flow: [B, 2, T, H, W] 光流
        
        Returns:
            output: [B, 16, T, H, W] 调制后的轨迹
        """
        B, C, T, H, W = traj_latent.shape
        
        # 1. 精炼光流
        refined_flow = self.flow_refine(flow)
        refined_flow = flow + refined_flow * control_scale
        
        # 2. 对每一帧进行 warp
        warped_frames = []
        for t in range(T):
            frame = traj_latent[:, :, t, :, :]  # [B, C, H, W]
            flow_t = refined_flow[:, :, t, :, :]  # [B, 2, H, W]
            
            # 使用光流进行 warp
            warped = self.warp_with_flow(frame, flow_t)
            warped_frames.append(warped)
        
        warped_traj = torch.stack(warped_frames, dim=2)  # [B, C, T, H, W]
        
        # 3. 特征增强
        enhanced = self.feature_enhance(traj_latent)
        
        # 4. 融合原始和 warped
        fused = torch.cat([warped_traj, enhanced], dim=1)
        output = self.fusion(fused)
        
        return output
    
    def warp_with_flow(self, x, flow):
        """
        使用光流对特征进行 warp
        
        Args:
            x: [B, C, H, W] 输入特征
            flow: [B, 2, H, W] 光流 (dx, dy)
        
        Returns:
            warped: [B, C, H, W] warp 后的特征
        """
        B, C, H, W = x.shape
        
        # 创建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # 应用光流
        grid = grid + flow
        
        # 归一化到 [-1, 1]
        grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
        
        # 转换为 grid_sample 格式 [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)
        
        # 执行 warp
        warped = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped

class TrajectoryConstrainedMotion(nn.Module):
    def __init__(self, traj_c=16, flow_c=2, hidden=128):
        super().__init__()
        
        # 1. 轨迹编码器（空间约束）
        self.traj_encoder = nn.Sequential(
            nn.Conv3d(traj_c, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        
        # 2. 光流编码器（运动信息）
        self.flow_encoder = nn.Sequential(
            nn.Conv3d(flow_c, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        
        # 3. 参考系对齐模块
        self.align_module = nn.Sequential(
            nn.Conv3d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
        )
        
        # 4. 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(hidden, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 5. 生成调制参数（修复：使用 traj_c 而不是 out_c）
        self.to_gamma = zero_module(nn.Conv3d(hidden, traj_c, 1))
        self.to_beta = zero_module(nn.Conv3d(hidden, traj_c, 1))
        self.norm = nn.GroupNorm(4, traj_c)  # 使用 traj_c
    
    def forward(self, traj_latent, flow, control_scale):
        """
        traj_latent: [B, 16, T, H, W] 静态视角的轨迹
        flow: [B, 2, T, H, W] 相机运动
        control_scale: 控制强度
        返回: [B, 16, T, H, W] 调制后的轨迹特征
        """
        # 编码
        traj_feat = self.traj_encoder(traj_latent)
        flow_feat = self.flow_encoder(flow)
        
        # 参考系对齐
        combined = torch.cat([traj_feat, flow_feat], dim=1)
        aligned = self.align_module(combined)
        
        # 空间注意力
        attention = self.spatial_attention(aligned)
        constrained = aligned * attention
        
        # 生成调制参数
        gamma = self.to_gamma(constrained)
        beta = self.to_beta(constrained)
        
        # FiLM 调制
        out = traj_latent + control_scale * (self.norm(traj_latent) * gamma + beta)
        
        return out

# ============================================光流===========================================

    def __init__(self, flow_c=2, hidden=64):
        super().__init__()
        
        # 1. 空间编码（保留位置信息）
        self.camera_spatial = nn.Sequential(
            nn.Conv3d(flow_c, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, hidden),
            nn.SiLU()
        )
        
        # 2. 时间建模
        self.camera_temporal = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, hidden),
            nn.SiLU()
        )
        
        # 3. 生成调制参数（不用 zero_module，用小初始化）
        self.to_gamma = nn.Conv3d(hidden, flow_c, 1)
        self.to_beta = nn.Conv3d(hidden, flow_c, 1)
        
        # 小初始化（而不是零初始化）
        nn.init.normal_(self.to_gamma.weight, std=0.01)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.normal_(self.to_beta.weight, std=0.01)
        nn.init.zeros_(self.to_beta.bias)
        
        # 4. 使用 GroupNorm（而不是 InstanceNorm）
        self.norm = nn.GroupNorm(1, flow_c)  # 1 group = LayerNorm
    
    def forward(self, static_object_flow, camera_flow, control_scale=1.0):
        # 空间+时间编码
        cam_feat = self.camera_spatial(camera_flow)
        cam_feat = self.camera_temporal(cam_feat)
        
        # 生成调制参数
        gamma = self.to_gamma(cam_feat)
        beta = self.to_beta(cam_feat)
        
        # FiLM 调制（改进版）
        normed = self.norm(static_object_flow)
        out = static_object_flow + control_scale * (normed * (1 + gamma) + beta)
        #                                                    ^^^^^^^^
        #                                                    关键：1 + gamma
        return out

    """
    专为稀疏光流设计的调制器
    """
    def __init__(self, flow_c=2, hidden=64):
        super().__init__()
        
        # 提取物体区域的相机光流
        self.camera_extractor = nn.Sequential(
            nn.Conv3d(flow_c, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, flow_c, 1)
        )
        
        # 小初始化
        nn.init.normal_(self.camera_extractor[-1].weight, std=0.01)
        nn.init.zeros_(self.camera_extractor[-1].bias)
    
    def forward(self, static_object_flow, camera_flow, control_scale=1.0):
        # 创建物体 mask
        mask = (static_object_flow.abs().sum(dim=1, keepdim=True) > 1e-4).float()
        
        # 提取物体区域的相机光流
        camera_at_object = self.camera_extractor(camera_flow) * mask
        
        # 简单相加
        output = static_object_flow + control_scale * camera_at_object
        
        return output
class WarpResidualFlowModulator(nn.Module):
    """
    将静态视角物体光流转换为动态视角物体光流。

    核心思路变更（v2）：
      - 旧方案：grid_sample 做空间 warp → 对稀疏光流几乎无效
      - 新方案：在物体区域做自适应加法融合 + 残差修正

    流程：
      1. 物体 mask 提取
      2. 在物体区域采样 camera_flow，与 static_flow 做自适应融合
      3. 残差网络修正融合误差（处理透视非均匀性、bbox 边界等）
      4. 输出 = fused_flow + residual

    输入：
      static_object_flow: [B, 2, T, H, W]  静态视角物体光流（稀疏）
      camera_flow:        [B, 2, T, H, W]  全局相机运动光流（稠密）
    输出：
      dynamic_object_flow: [B, 2, T, H, W] 动态视角物体光流
    """

    def __init__(self, hidden=32, num_res_blocks=2):
        super().__init__()

        # 自适应融合权重网络
        # 输入：static_flow(2) + camera_flow(2) + mask(1) = 5
        # 输出：融合权重 alpha(1)，控制 camera_flow 加多少
        self.fusion_net = nn.Sequential(
            nn.Conv3d(5, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),  # alpha in [0, 1]
        )

        # 残差修正网络
        # 输入：fused_flow(2) + camera_flow(2) + mask(1) = 5
        in_c = 5
        layers = [
            nn.Conv3d(in_c, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(hidden))
        layers.append(nn.Conv3d(hidden, 2, kernel_size=1))
        self.residual_net = nn.Sequential(*layers)

        # 残差网络输出初始化接近零
        nn.init.normal_(self.residual_net[-1].weight, std=0.001)
        nn.init.zeros_(self.residual_net[-1].bias)

    def forward(self, static_object_flow, camera_flow, control_scale=1.0):
        """
        static_object_flow: [B, 2, T, H, W]
        camera_flow:        [B, 2, T, H, W]
        """
        # 1. 物体 mask
        mask = (static_object_flow.abs().sum(dim=1, keepdim=True) > 1e-4).float()

        # 2. 自适应融合：alpha 控制 camera_flow 的混合比例
        fusion_input = torch.cat([static_object_flow, camera_flow, mask], dim=1)
        alpha = self.fusion_net(fusion_input)  # [B, 1, T, H, W]

        # 融合：static + alpha * camera，只在物体区域
        fused_flow = static_object_flow + alpha * camera_flow * mask

        # 3. 残差修正
        residual_input = torch.cat([fused_flow, camera_flow, mask], dim=1)
        residual = self.residual_net(residual_input)
        residual = residual * mask  # 背景保持零

        # 4. 输出
        output = fused_flow + control_scale * residual

        return output


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

    def forward(self, x):
        h = self.spatial(x)
        h = self.temporal(h)
        return x + h


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
        self.flow_modulator = ImprovedFlowWarpModulator(
            traj_c=cin,
            flow_c=2,
            hidden=128
        )
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
        # zero_module(self.flow_modulator.to_gamma)
        # zero_module(self.flow_modulator.to_beta)

    def forward(self, traj_latent, flow, warmup_scale = 1.0):
        """
        x: torch.Tensor: shape [B C T H W]
        """
        # 移除 detach：让 FiLM 参数能通过 predicted_flow 的梯度学习
        # AuxHead 已冻结，不会被更新
        x = self.flow_modulator(traj_latent, flow, warmup_scale)
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
