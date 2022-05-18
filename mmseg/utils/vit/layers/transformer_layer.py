import torch
import torch.nn as nn
from mmseg.utils.vit.blocks.attention import Attention
from mmseg.utils.vit.blocks.mlp import MLP


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nof_heads, mlp_ratio=4., qkv_bias=False, qk_scale = None, drop = 0., attn_drop=0.,
                 drop_path = 0., act_layer=nn.GELU, norm_layer = nn.LayerNorm):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, nof_heads=nof_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop = attn_drop, proj_drop=drop)

        self.dropt_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(input_features=d_model, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attn=False):
        y, attn = self.attn(self.norm1(x))
        if return_attn:
            return attn
        x = x + self.dropt_path(y)
        x = x + self.dropt_path(self.mlp(self.norm2(x)))
        return x



