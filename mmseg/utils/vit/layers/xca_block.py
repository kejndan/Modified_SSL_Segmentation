import torch
import torch.nn as nn
from mmseg.utils.vit.blocks.xca_attention import XCA
from mmseg.utils.vit.layers.transformer_layer import DropPath
from mmseg.utils.vit.blocks.mlp import MLP
from mmseg.utils.vit.blocks.lpi import LPI


class XCABlock(nn.Module):
    def __init__(self, d_model, nb_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tokens=196, eta=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = XCA(
            d_model=d_model, nb_heads=nb_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(input_features=d_model, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        self.norm3 = nn.LayerNorm(d_model)
        self.local_mp = LPI(in_features=d_model, act_layer=act_layer)

        self.gamma1 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x
