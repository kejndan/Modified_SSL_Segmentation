import torch
from torch import nn


class PEG(nn.Module):
    def __init__(self, config, d_model, k=3):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(d_model, d_model, k, 1, k//2, groups=d_model)
        # Only for demo use, more complicated functions are effective too.

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.config.output_encoder == 'CLS':
            cls_token, feat_token = x[:, 0], x[:, 1:]
            cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        elif self.config.output_encoder == 'GAP':
            feat_token = x
            cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat
            x = x.flatten(2).transpose(1, 2)
        return x

