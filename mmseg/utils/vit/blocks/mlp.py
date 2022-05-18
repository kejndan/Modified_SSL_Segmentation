import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_features, hidden_features=None, out_features = None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or input_features
        hidden_features = hidden_features or input_features

        self.fc1 = nn.Linear(input_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

