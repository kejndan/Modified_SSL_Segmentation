import torch
import torch.nn as nn


class PathEmbedding(nn.Module):
    """"""
    """
    Image -> Patch Embedding.
    
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embedding_dim=768, freeze_proj=False, norm_layer=None):
        super().__init__()
        self.numb_patches = (img_size // patch_size) * (img_size // patch_size)
        self.image_size = img_size
        self.patch_size = patch_size

        self.linear_transform = nn.Conv2d(in_chans, embedding_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embedding_dim)
        else:
            self.norm = None

        if freeze_proj:
            self.linear_transform.weight.requires_grad = False
            self.linear_transform.bias.requires_grad = False

    def forward(self, x):
        """ 
        :param x: [B, C, H, W]
        :return: [B, N, E], N - number of patches, E - embedding dim
        """""
        x = self.linear_transform(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
                  ),
        nn.BatchNorm2d(out_planes)
    )


class ConvPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=192):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        nb_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.nb_patches = nb_patches

        if patch_size[0] == 16:
            self.proj = nn.Sequential(
                conv3x3(3, d_model // 8, 2),
                nn.GELU(),
                conv3x3(d_model // 8, d_model // 4, 2),
                nn.GELU(),
                conv3x3(d_model // 4, d_model // 2, 2),
                nn.GELU(),
                conv3x3(d_model // 2, d_model, 2),
            )
        elif patch_size[0] == 8:
            self.proj = nn.Sequential(
                conv3x3(3, d_model // 4, 2),
                nn.GELU(),
                conv3x3(d_model // 4, d_model // 2, 2),
                nn.GELU(),
                conv3x3(d_model // 2, d_model, 2),
            )

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)



