import torch
import torch.nn as nn




class Attention(nn.Module):
    """"""
    """
    Multi Head Attention mechanism.
    """

    def __init__(self, d_model, nof_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.nof_heads = nof_heads
        self.head_dim = d_model // nof_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)


    def multi_q_k_product(self, x, B, N):
        """
        :return Non-softmax attenttion [batch_size, nb_heads, nb_tokens, nb_tokens];
                Values [batch_size, nb_heads, nb_tokens, head_dim]
        """
        qkv = self.qkv(x).reshape(B, N, 3, self.nof_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, nof_heads,
        # N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        return attn, v

    def forward(self, x):
        B, N, C = x.shape
        # unlike base notation, uses 1 linear projection

        attn, values = self.multi_q_k_product(x, B, N)
        attention = attn.softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ values).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attention


if __name__ == '__main__':
    s = Attention(10)
    print(s)
