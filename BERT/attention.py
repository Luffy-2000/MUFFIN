import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask==0, 1e-9)
        attn = F.softmax(attn, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_headed, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = num_headed
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.w_ks = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.w_vs = nn.Linear(self.d_model, self.n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_v)))

        self.attention = Attention()
        self.fc = nn.Linear(self.n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask = None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        #residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask, dropout=self.dropout)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.fc(output)
        return output

if __name__ == "__main__":
    num_headed = 8
    d_model = 128
    d_k = 64
    d_v = 64
    a = MultiHeadAttention(num_headed, d_model, d_k, d_v)
    x = torch.randn(64, 16, 128)
    print(a(x,x,x).shape)

