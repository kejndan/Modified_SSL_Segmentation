import torch
from torch import nn
from torch import Tensor
import numpy as np

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



class SinCosPositionEncoding(torch.nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    # @torch.jit.script
    def linspace(self, start: Tensor, stop: Tensor, num: int):
        """
        Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
        Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
        """
        # create a tensor of 'num' steps from 0 to 1
        steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

        # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
        # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
        #   "cannot statically infer the expected size of a list in this contex", hence the code below
        for i in range(start.ndim):
            steps = steps.unsqueeze(-1)

        # the output starts at 'start' and increments until 'stop' in each dimension
        out = start[None] + steps * (stop - start)[None]

        return out.T

    def forward(self, input, info_crop):
        batch_size, nb_h, nb_w, d_model = input.size()
        device = input.device

        x = torch.zeros([batch_size, nb_h, nb_w]).to(device)
        y = torch.zeros([batch_size, nb_h, nb_w]).to(device)

        x += torch.linspace(-1, 1, nb_h)[None, :, None].to(device)
        y += torch.linspace(-1, 1, nb_w)[None, None, :].to(device)

        half_d_model = d_model // 2
        rho = 10 ** (torch.arange(1, half_d_model + 1) / half_d_model).to(device)
        w_x = rho * torch.cos(torch.arange(half_d_model)).to(device)
        w_y = rho * torch.sin(torch.arange(half_d_model)).to(device)

        phase = np.pi * (w_x * x[:, :, :, None] + w_y * y[:, :, :, None]).to(device)
        pos_emb = torch.cat((torch.cos(phase), torch.sin(phase)), axis=-1)

        return pos_emb
