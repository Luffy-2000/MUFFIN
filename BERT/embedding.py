import torch.nn as nn
import torch
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(self.max_len, self.d_model).float()
        pe.require_grad = False

        position  = torch.arange(0, self.max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0)/self.d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,x):
        assert x.size(1) <=  self.max_len, "sequence length must be no greater than max_len"
        return self.pe[:, :x.size(1)]

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, embed_size, max_len, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.embed_size = embed_size//2
        self.token1 = TokenEmbedding(vocab_size = vocab_size1, embed_size = self.embed_size)
        self.token2 = TokenEmbedding(vocab_size = vocab_size2, embed_size = self.embed_size)
        self.position = PositionalEmbedding(d_model=embed_size, max_len= max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x1, x2):
        # x.shape batch_size * sequence_length
        assert x1.size(1) <= self.max_len, "sequence length must be no greater than max_len"
        assert x2.size(1) <= self.max_len, "sequence length must be no greater than max_len"
        #x = self.token1(x1) + self.token2(x2) + self.position(x1)
        x = torch.cat((self.token1(x1), self.token2(x2)),dim=-1) + self.position(x1)
        return x


if __name__ == "__main__":
    #vocab_size = 1500
    #max_len 所用包数
    #d_model = embedding_size 隐藏层参数
    bert_emd = BERTEmbedding(vocab_size1=1501, vocab_size2=65535, embed_size=1024, max_len=16)
    x1 = torch.LongTensor([[1500,1400, 0, 1]])
    x2 = torch.LongTensor([[1282, 45, 12313, 0]])
    #x = torch.zeros(128, 1500)
    #print(x.shape)
    print(bert_emd(x1, x2).shape)