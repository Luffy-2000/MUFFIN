import torch.nn as nn
from BERT.transformer import TransformerBlock
from BERT.embedding import BERTEmbedding

class BERT(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, max_len, hidden=768, n_layers=12, attn_heads=12, d_k=64, d_v=64, dropout=0.1):
        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4
        self.embedding = BERTEmbedding(vocab_size1=vocab_size1, vocab_size2=vocab_size2,
                                       embed_size=hidden, max_len=max_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(h=attn_heads, hidden=hidden, d_v=d_k, d_k=d_v,
                              feed_forward_hidden=hidden * 4, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self,x1,x2):
        #mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x1,x2)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, None)
        return x