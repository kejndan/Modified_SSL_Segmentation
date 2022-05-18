import torch
from mmseg.utils.vit.layers.swin_block import SwinTransformerBlock
from torch import nn

class SwinLayer(SwinTransformerBlock):
    def __init__(self, d_model, input_resolution, nb_blocks, nb_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 ):
        super().__init__()