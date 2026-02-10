import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class MultiFreqFlowHead(nn.Module):
    def __init__(self, in_dim=3072,
                 latent_channels=16,
                 patch_size = 2,
                 out_latent_frames=13,
                 latent_h = 60,
                 latent_w = 90):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.out_latent_frames = out_latent_frames
        self.latent_h = latent_h
        self.latent_w = latent_w

        self.flow_proj = nn.Linear(in_dim, patch_size * patch_size * latent_channels)

        self.num_video_tokens = (
            out_latent_frames * (latent_h // patch_size) * (latent_w // patch_size)
        )

        self.temporal_upsample = nn.ConvTranspose3d(
            latent_channels, latent_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0)
        )
        # 低频头
        self.flow_low = nn.Sequential(
            nn.Conv3d(latent_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(),
            nn.Conv3d(16, 2, kernel_size=3, padding=1)
        )

        # 高频头
        self.flow_high = nn.Sequential(
            nn.Conv3d(latent_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(),
            nn.Conv3d(32, 2, kernel_size=3, padding=1)
        )

    def forward(self, feat_tokens):
        # 4. Final block
        hidden_states = self.flow_proj(feat_tokens)
        B, _, _ = hidden_states.shape

        # 5. Unpatchify
        latent = hidden_states.reshape(B, self.out_latent_frames, self.latent_h // self.patch_size, self.latent_w // self.patch_size, -1, self.patch_size, self.patch_size)
        latent = latent.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        # 插值/ 时间上采样 T=13->T=24
        x = latent.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        # flow = F.interpolate(x, size=(24, 60, 90), mode="trilinear", align_corners=False)
        flow = self.temporal_upsample(x)  # output: [B, C, 25, H, W]
        flow = flow[:, :, :24]  # 截取前 24 帧

        # 3. 包 checkpoint → 激活值不保存
        pred_low = checkpoint(self.flow_low, flow, use_reentrant=False)
        pred_high = checkpoint(self.flow_high, flow, use_reentrant=False)

        return pred_low, pred_high

from torch.utils.tensorboard import SummaryWriter

import torch
if __name__ == '__main__':
    # 创建模型和输入张量
    model = MultiFreqFlowHead()
    input_tensor = torch.rand(1,17550,3072)

    # 初始化SummaryWriter
    writer = SummaryWriter("tensorboard")
    # 将模型和输入张量添加到TensorBoard
    writer.add_graph(model, input_tensor)
    # 关闭SummaryWriter
    writer.close()