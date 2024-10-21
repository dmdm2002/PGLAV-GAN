import torch
import torch.nn as nn

from local_attention import LocalMHA


class CAM(nn.Module):
    '''Construction of the channel attention'''

    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (B,C,H,W)-->(B,C,1,1)
        # self.avg_pool: Compute spatial average value
        # self.max_pool: Compute spatial maximum value
        # Sharing MLP layers
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)

        return x * out


class ReAttention(nn.Module):
    """
    It is observed that similarity along same batch of data is extremely large.
    Thus can reduce the bs dimension when calculating the attention map.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expansion_ratio=3,
                 apply_transform=True, transform_scale=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.apply_transform = apply_transform

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
            self.var_norm = nn.BatchNorm2d(self.num_heads)
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, atten=None):
        B, N, C = x.shape
        # x = self.fc(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


class ParallelAttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7):
        super(ParallelAttentionModule, self).__init__()

        self.re_attn = ReAttention(dim=dim, num_heads=num_heads)
        self.local_attn = LocalMHA(dim=dim, dim_head=dim // 8, window_size=window_size, causal=True)

    def forward(self, x):
        re_attn_y, _ = self.re_attn(x)
        local_attn_y = self.local_attn(x)

        return re_attn_y + local_attn_y


class SequenceAttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7):
        super(SequenceAttentionModule, self).__init__()

        self.re_attn = ReAttention(dim=dim, num_heads=num_heads)
        self.local_attn = LocalMHA(dim=dim, window_size=window_size, causal=True)

    def forward(self, x, re_alpha):
        local_attn_y = self.local_attn(x)

        attn_y_1 = x + re_alpha * local_attn_y

        re_attn_y, _ = self.re_attn(attn_y_1)

        return re_attn_y

