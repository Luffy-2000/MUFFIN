import torch
import torch.nn as nn
from BERT.attention import MultiHeadAttention
from BERT.utils import SublayerConnection, PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, h, hidden, d_k, d_v, feed_forward_hidden, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_headed=h, d_model=hidden, d_k=d_k, d_v=d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x,_x,_x,mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

if __name__ == "__main__":
    h = 8
    hidden = 512
    d_k = 64
    d_v = 64
    feed_forward_hidden = 1024
    dropout = 0.1
    transformer = TransformerBlock(h,hidden,d_k,d_v,feed_forward_hidden,dropout)
    input = torch.randn(64, 16, 512)
    print(transformer(input, None).shape)
