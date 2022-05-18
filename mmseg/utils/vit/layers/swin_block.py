import torch
from torch import nn
from mmseg.utils.vit.blocks.window_attention import WindowAttention
from mmseg.utils.vit.blocks.mlp import MLP
from mmseg.utils.vit.swin_utils.window_operations import window_partition, window_reverse
from mmseg.utils.vit.layers.transformer_layer import DropPath

class SwinTransformerBlock(nn.Module):
    def __init__(self, d_model, input_resolution, nb_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.input_resolution = input_resolution
        self.nb_heads = nb_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio


        self.norm1 = norm_layer(d_model)
        self.attn = WindowAttention(d_model, window_size=(self.window_size, self.window_size), nb_heads=nb_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, atth_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(self.d_model)
        mlp_hidden_dim = int(self.d_model * mlp_ratio)
        self.mlp = MLP(input_features=self.d_model, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))


            counter = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = counter
                    counter += 1

            mask_windows = window_partition(img_mask, self.window_size) # nb_w, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, N, d_model = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, d_model)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size) #nb_w, window_size, window_size, d_model
        x_windows = x_windows.view(-1, self.window_size * self.window_size, d_model)


        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse window cycle
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, d_model)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


if __name__ == '__main__':
    swin = SwinTransformerBlock(192, (63, 63),3)
    swin.forward(torch.rand(5, 3969, 192))

