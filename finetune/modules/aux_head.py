import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1. Frame Encoder（和你原来几乎一样）
# ------------------------------------------------------------
class FrameEncoder(nn.Module):
    def __init__(self, in_ch=16, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 2, 1),   # 60 -> 30
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1),  # 30 -> 15
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)   # [B, 128, 15, 23]


# ------------------------------------------------------------
# 2. Pairwise Flow Head（核心变化）
# ------------------------------------------------------------
class FlowHead(nn.Module):
    """
    输入: concat(f_{t-1}, f_t)  [B,256,15,23]
    输出: flow (downsampled)    [B,2,15,23]
    """
    def __init__(self, in_ch=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, 1, 1),
        )

        # 关键：初始化为「预测 0 flow」
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# 3. AuxHead（最终版本）
# ------------------------------------------------------------
class AuxHead(nn.Module):
    """
    Input : latent video [B,T,16,60,90]
    Output: flow         [B,T, 2,60,90]
    flow[:,0] = 0
    flow[:,t] = motion(t-1 -> t)
    """
    def __init__(self):
        super().__init__()
        self.encoder = FrameEncoder()
        self.flow_head = FlowHead()

    def forward(self, latent):
        # latent: [B,T,16,60,90]
        # latent = latent.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C, H, W = latent.shape
        device = latent.device

        flows = []

        # t = 0 → zero flow
        flows.append(torch.zeros(B, 2, H, W, device=device))

        # encode all frames
        feats = []
        for t in range(T):
            feats.append(self.encoder(latent[:, t]))  # [B,128,15,23]

        # predict pairwise flow
        for t in range(1, T):
            f_prev = feats[t - 1]
            f_cur  = feats[t]

            x = torch.cat([f_prev, f_cur], dim=1)  # [B,256,15,23]
            flow_low = self.flow_head(x)            # [B,2,15,23]

            # 上采样回 latent 分辨率
            flow = F.interpolate(
                flow_low,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

            flows.append(flow)

        flows = torch.stack(flows, dim=1)  # [B,T,2,60,90]
        return flows

from torch.utils.tensorboard import SummaryWriter

import torch
if __name__ == '__main__':
    # 创建模型和输入张量
    model = AuxHead()
    input_tensor = torch.rand(1, 13, 16, 60, 90)
    # 初始化SummaryWriter
    writer = SummaryWriter("tensorboard")
    # 将模型和输入张量添加到TensorBoard
    writer.add_graph(model, input_tensor)
    # 关闭SummaryWriter
    writer.close()