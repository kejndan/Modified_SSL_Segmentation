import torch
import torch.nn as nn
from mmseg.utils.vit.layers.transformer_layer import DropPath
from mmseg.utils.vit.blocks.mlp import MLP

class XCA(nn.Module):
    def __init__(self, d_model, nb_heads=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.nb_heads = nb_heads
        self.temperature = nn.Parameter(torch.ones(nb_heads, 1, 1))

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nb_heads, C // self.nb_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class ClassAttention(nn.Module):
    def __init__(self, d_model, nb_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.nb_heads = nb_heads
        head_dim = d_model // nb_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nb_heads, C // self.nb_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qc = q[:, :, 0:1]
        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
        cls_tkn = self.proj(cls_tkn)

        x = torch.cat([self.proj_drop(cls_tkn), x[:, 1:]], dim=1)

        return x


class ClassAttentionBlock(nn.Module):
    def __init__(self, d_model, nb_heads, mlp_ratio = 4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=None,
                 tokens_norm=False):
        super().__init__()

        self.norm1 = norm_layer(d_model)

        self.attn = ClassAttention(d_model=d_model, nb_heads=nb_heads,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = MLP(input_features=d_model, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        if eta is not None:
            self.gamma1 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        self.tokens_norm = tokens_norm

    def forward(self, x, H, W, mask=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x[:, 0:1]  = self.norm2(x[:, 0:1])

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x
