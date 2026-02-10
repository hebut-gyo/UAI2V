import torch.nn as nn
import torch
from torch.nn.init import trunc_normal_
class MediatorAttention(nn.Module):
    def __init__(self, dim, latent_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, mediator_num=64,patch_size=2, **kwargs):
        super().__init__()
        self.dim = dim
        self.latent_size = latent_size  # Wh, Ww
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size

        self.mediator_num = mediator_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, mediator_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, mediator_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, mediator_num, latent_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, mediator_num, 1, latent_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, latent_size[0], 1, mediator_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, latent_size[1], mediator_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(mediator_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def _token_grid(self):
        H_lat, W_lat = self.latent_size
        Ht = H_lat // self.patch_size
        Wt = W_lat // self.patch_size
        return Ht, Wt
    def forward(self, x, mask=None):
        B, N, C = x.shape
        # h = int(n ** 0.5) n = h*w
        # w = int(n ** 0.5)
        Ht, Wt = self._token_grid()
        HW = Ht * Wt
        T = N // HW
        x_bt = x.view(B, T, HW, C).reshape(B * T, HW, C) # x:(B N C)->(B*T HW C) 从N中分离T维度并将其与B合并
        # qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        # q, k, v: b, n, c
        qkv = self.qkv(x_bt).reshape(B * T, HW, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # q, k, v: B*T, HW, C
        # mediator_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        # (B*T HW C)->(B*T Ht Hw C)->(B*T C Ht Hw)->池化->(B*T C M)->(B*T M C)
        mediator_tokens = self.pool(q.reshape(B * T, Ht, Wt, C).permute(0, 3, 1, 2)).reshape(B * T, C, -1).permute(0, 2, 1)

        num_heads = self.num_heads
        head_dim = C // num_heads
        q = q.reshape(B * T, HW, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * T, HW, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * T, HW, num_heads, head_dim).permute(0, 2, 1, 3)

        mediator_tokens = mediator_tokens.reshape(B * T, self.mediator_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # position_bias1 = nn.functional.interpolate(self.an_bias, size=self.latent_size, mode='bilinear')
        # position_bias1 = position_bias1.reshape(1, num_heads, self.mediator_num, -1).repeat(b, 1, 1, 1)
        # position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.mediator_num, -1).repeat(b, 1, 1, 1)
        # position_bias = position_bias1 + position_bias2
        # mediator_attn = self.softmax((mediator_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        # mediator_attn = self.attn_drop(mediator_attn)
        # mediator_v = mediator_attn @ v
        #
        # mediator_bias1 = nn.functional.interpolate(self.na_bias, size=self.latent_size, mode='bilinear')
        # mediator_bias1 = mediator_bias1.reshape(1, num_heads, self.mediator_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1,
        #                                                                                                         1)
        # mediator_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.mediator_num).repeat(b, 1, 1, 1)
        # mediator_bias = mediator_bias1 + mediator_bias2
        # q_attn = self.softmax((q * self.scale) @ mediator_tokens.transpose(-2, -1) + mediator_bias)
        # q_attn = self.attn_drop(q_attn)
        # x = q_attn @ mediator_v
        # x = x.transpose(1, 2).reshape(b, n, c)
        # v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # position bias: an_bias (h, M, 7,7) -> (h, M, Ht, Wt) -> (BT, h, M, HW)
        pos_bias = nn.functional.interpolate(self.an_bias, size=(Ht, Wt), mode="bilinear", align_corners=False)
        pos_bias = pos_bias.reshape(1, num_heads, self.mediator_num, HW).repeat(B * T, 1, 1, 1)

        # mediator -> tokens
        mediator_attn = self.softmax((mediator_tokens * self.scale) @ k.transpose(-2, -1) + pos_bias)  # (BT, h, M, HW)
        mediator_attn = self.attn_drop(mediator_attn)

        mediator_v = mediator_attn @ v  # (BT, h, M, d)

        # tokens -> mediator bias: na_bias (h, M, 7,7) -> (h, M, Ht,Wt) -> (BT,h,HW,M)
        med_bias = nn.functional.interpolate(self.na_bias, size=(Ht, Wt), mode="bilinear", align_corners=False)
        med_bias = med_bias.reshape(1, num_heads, self.mediator_num, HW).permute(0, 1, 3, 2).repeat(B * T, 1, 1, 1)

        q_attn = self.softmax((q * self.scale) @ mediator_tokens.transpose(-2, -1) + med_bias)  # (BT, h, HW, M)
        q_attn = self.attn_drop(q_attn)

        x = q_attn @ mediator_v  # (BT, h, HW, d)
        # merge heads: (BT, HW, C)
        x = x.permute(0, 2, 1, 3).reshape(B * T, HW, C)

        # depthwise conv residual on token grid
        v_map = v.permute(0, 2, 1, 3).reshape(B * T, Ht, Wt, C).permute(0, 3, 1, 2)  # (BT, C, Ht, Wt)

        x = x + self.dwc(v_map).permute(0, 2, 3, 1).reshape(B * T, HW, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # back to (B, N, C)
        x = x.view(B, T, HW, C).reshape(B, N, C)

        return x

class MediatorAttentionWrapper(nn.Module):
    def __init__(self, base_attn, latent_size, patch_size):
        super().__init__()
        self.base_attn = base_attn  # 3D Full Attention
        self.mediator_attn = MediatorAttention(
            dim=base_attn.query_dim,
            latent_size=latent_size,  # 空间尺寸
            num_heads=base_attn.heads,
            mediator_num=64,
            patch_size=patch_size,
        )
        # 门控参数，初始为 0
        self.alpha = nn.Parameter(torch.tensor(1e-3))
        self._aligned = False

    @torch.no_grad()
    def _maybe_align(self, hidden_states):
        if self._aligned:
            return
        device = hidden_states.device
        dtype = hidden_states.dtype
        self.mediator_attn.to(device=device, dtype=dtype)
        self.alpha.data = self.alpha.data.to(device=device, dtype=dtype)
        self._aligned = True
    def forward(self, hidden_states, **kwargs):
        # 关键：先对齐
        self._maybe_align(hidden_states)
        out_hidden, out_encoder = self.base_attn(hidden_states, **kwargs)
        out_mediator = self.mediator_attn(hidden_states)
        out_hidden = out_hidden + self.alpha * out_mediator
        return out_hidden, out_encoder