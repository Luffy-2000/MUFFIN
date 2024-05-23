import torch.nn as nn
from BERT.bert import BERT

class BERTLM(nn.Module):
    def __init__(self, bert:BERT, vocab_size1, vocab_size2):
        super(BERTLM, self).__init__()
        self.bert = bert
        self.mask_lm1 = MaskedLanguageModel(self.bert.hidden//2, vocab_size1)
        self.mask_lm2 = MaskedLanguageModel(self.bert.hidden//2, vocab_size2)

    def forward(self,x1,x2):
        x = self.bert(x1,x2)
        x_pkt = x[:,:, :self.bert.hidden//2]
        x_win = x[:,:, self.bert.hidden//2:]
        return self.mask_lm1(x_pkt), self.mask_lm2(x_win)


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

