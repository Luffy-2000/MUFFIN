import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class Comparison_Net(nn.Module):
    def __init__(self, seq_len, hidden_size, dropout=0.1):
        super(Comparison_Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=2*seq_len, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(int(hidden_size/8)*16, 1)
    def forward(self,x1,x2):
        x = torch.cat([x1,x2],dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.shape[0],-1)
        seq_score = torch.sigmoid(self.fc1(x))
        return seq_score


class FE_embedding(nn.Module):
    def __init__(self, v1, v2, hidden):
        super(FE_embedding, self).__init__()
        self.e1 = nn.Embedding(v1, hidden)
        self.e2 = nn.Embedding(v2, hidden)

    def forward(self, s1, s2):
        return self.e1(s1) + self.e2(s2)



if __name__ == "__main__":
    input1 = torch.randn(1, 4, 1024)
    input2 = torch.randn(1, 4, 1024)
    c_net = Comparison_Net(4, 1024)
    output1 = c_net(input1, input2)
    output2 = c_net(input2, input1)
    print(output1)
    print(output2)

