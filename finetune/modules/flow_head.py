import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class MultiFreqFlowHead(nn.Module):
    def __init__(self, in_dim=3072, latent_channels=16):
        super().__init__()
        self.latent_channels = latent_channels

        # token -> latent 共享投影
        self.proj = nn.Linear(in_dim, latent_channels) # 3072->16

        # 低频头
        self.flow_low = nn.Sequential(
            nn.Conv3d(latent_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 2, kernel_size=3, padding=1)
        )

        # 高频头
        self.flow_high = nn.Sequential(
            nn.Conv3d(latent_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 2, kernel_size=3, padding=1)
        )

    def forward(self, feat_tokens):
        feat_tokens = feat_tokens.to(dtype=self.proj.weight.dtype)
        B, N, D = feat_tokens.shape

        # 投影并 reshape
        x = self.proj(feat_tokens)          # [B, 17750, 16]
        x = x.permute(0, 2, 1)              # [B, 16, 17750]

        x = x.view(B, self.latent_channels, -1, 1, 1)  # [B,C,N,1,1]
        x = F.interpolate(x, size=(13, 60, 90),
                          mode='trilinear', align_corners=False)# [B,16,13,60,90]
        # 上采样到视频原始尺寸
        x_up = F.interpolate(x, size=(24, 60, 90),
                                    mode='trilinear', align_corners=False)# [B,16,48,480,720]

        # 3. 包 checkpoint → 激活值不保存
        pred_low = checkpoint(self.flow_low, x_up, use_reentrant=False)
        pred_high = checkpoint(self.flow_high, x_up, use_reentrant=False)
        return pred_low, pred_high