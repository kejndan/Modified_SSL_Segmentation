import torch
import torch.nn as nn



class LPI(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop = 0., kernel_size=3):
        super().__init__()
        out_features  = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                               padding=padding, groups=out_features)
        self.act = act_layer()

        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                               padding=padding, groups=out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x

