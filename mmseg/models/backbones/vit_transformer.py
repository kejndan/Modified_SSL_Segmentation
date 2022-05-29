import torch
import torch.nn as nn
import math
from mmseg.utils.vit.layers.transformer_layer import TransformerBlock
from mmseg.utils.vit.blocks.tokenizer import PathEmbedding
from functools import partial
from mmseg.utils.vit.scripts.trunc_normal import trunc_normal_
from mmseg.utils.vit.blocks.position_encoding import PEG, SinCosPositionEncoding
from ..builder import BACKBONES
from torch.nn.modules.utils import _pair as to_2tuple
# See also: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

@BACKBONES.register_module()
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None,
                 qkv_scale=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        if isinstance(img_size, tuple):
            img_size = img_size[0]
        self.d_model = embed_dims
        self.out_indices = out_indices
        self.patch_embed = PathEmbedding(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_channels, embedding_dim=self.d_model, freeze_proj=False)
        num_patches = self.patch_embed.numb_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.patch_size = patch_size
        self.pos_encoding = 'sincos'
        self.output_encoder = 'CLS'

        if self.pos_encoding == 'none':
            if self.output_encoder == 'GAP':
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.d_model))
            elif self.output_encoder == 'CLS':
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.d_model))
            trunc_normal_(self.pos_embed, std=.02)
        elif self.pos_encoding == 'sincos':
            self.pos_block = SinCosPositionEncoding(self.patch_size)

        # elif self.pos_encoding == 'PEG':
        #     self.pos_block = PEG(config, self.d_model)


        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=embed_dims, nof_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qkv_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                             norm_layer=norm_layer)
            for i in range(num_layers)])
        self.norm = norm_layer(self.d_model)

        # self.head = nn.Linear(self.d_model,
        #                       config.head_projection_out_dim) if config.head_projection_out_dim > 0 else nn.Identity()
        self.head = nn.Identity()
        # add trunc_normal for cls and pos_emb

        trunc_normal_(self.cls_token, std=0.2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # add a small number to avoid floating point error in the interpolation
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic', align_corners=True
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_token(self, x, shifts):
        B, nc, w, h = x.shape

        x = self.patch_embed(x)

        if self.output_encoder == 'CLS':
            cls_token = self.cls_token.expand(B, -1, -1)

            x = torch.cat((cls_token, x), dim=1)

        if self.pos_encoding == 'none':
            x = x + self.interpolate_pos_encoding(x, w, h)
        elif self.pos_encoding == 'sincos':
            reshaped_tokens = x[:,1:].reshape(-1, h // self.patch_embed.patch_size, w // self.patch_embed.patch_size, self.d_model)
            x[:, 1:] += self.pos_block(reshaped_tokens, shifts).reshape(B, -1, self.d_model)


        return self.pos_drop(x), w, h

    def forward2(self, x):
        x, w, h = self.prepare_token(x)

        patched_w, patched_h = w // self.patch_embed.patch_size, h // self.patch_embed.patch_size
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.pos_encoding == 'PEG':
                x = self.pos_block(x, patched_w, patched_h)
        x = self.norm(x)

        if self.output_encoder == 'CLS':
            x = x[:, 0]
        elif self.output_encoder == 'GAP':
            x = self.avgpool(x.transpose(1, 2))
            x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward(self, x, shifts=None):
        x, w, h = self.prepare_token(x, shifts)

        outs = []
        patched_w, patched_h = w // self.patch_embed.patch_size, h // self.patch_embed.patch_size
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.pos_encoding == 'PEG':
                x = self.pos_block(x, patched_w, patched_h)

            if i in self.out_indices:

                x = self.norm(x)
                out = x[:, 1:]
                B, _, C = out.shape

                out = out.reshape(B, patched_w, patched_h,
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs



    def get_last_selfattention(self, x):
        x, w, h = self.prepare_token(x)
        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = block(x)
            else:
                return block(x, return_attn=True)

    def get_intermediate_layers(self, x, n=1):
        # return the token output from the n last blocks
        x, w, h = self.prepare_token(x)
        output = []
        for i, block in self.blocks:
            x = block(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

# if __name__ == '__main__':
#     model = VisionTransformer(vit_config)
#     print(model)
#     print(f'Total param', sum(p.numel() for p in model.parameters()))

