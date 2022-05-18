import torch
import torch.nn as nn
from mmseg.utils.vit.blocks.attention import Attention
from timm.models.layers import trunc_normal_


class WindowAttention(Attention):
    def __init__(self, d_model, window_size, nb_heads = 8, qkv_bias = False, qk_scale = None, atth_drop=0., proj_drop=0.
                 ):
        super(WindowAttention, self).__init__(d_model, nb_heads, qkv_bias, qk_scale, atth_drop, proj_drop)
        self.window_size = window_size
        self.nof_heads = nb_heads
        self.head_dim = d_model // self.nof_heads
        self.scale = qk_scale or self.head_dim

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), self.nof_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]# 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1]-1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(atth_drop)
        self.proj = nn.Linear(d_model, d_model)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)



    def applying_mask(self, attn, mask, B, N):
        nb_windows = mask.shape[0]
        attn = attn.view(B // nb_windows, nb_windows, self.nof_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.nof_heads, N, N)

        return attn

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.nof_heads, C // self.nof_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        print(relative_position_bias.unsqueeze(0).size())

        print(attn.size())

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = self.applying_mask(attn, mask, B_, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



if __name__ == '__main__':
    wa = WindowAttention(192, (7, 7), 2)
    x = torch.rand(32, 49, 192)
    wa(x)