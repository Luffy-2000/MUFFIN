import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from data_processsing_new import DataManager


class BERTDataset(Dataset):
    def __init__(self, data, vocab_size1, vocab_size2,  masked=True):
        super(BERTDataset, self).__init__()
        max_len = int(data.shape[-1] / 2)
        self.x1 = torch.from_numpy(data[:, 1:max_len+1])
        self.x2 = torch.from_numpy(data[:, max_len+1:])
        self.y = torch.from_numpy(data[:, 0])
        #self.masked_ratio = masked_ratio
        self.MASK = 2
        self.vocab_size1 = vocab_size1
        self.vocab_size2 = vocab_size2
        self.masked = masked
    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.masked == True:
            masked_pkt_seq, masked_win_seq, pkt_masked_label, win_masked_label= self.random_pkt_seq(self.x1[item], self.x2[item])
            return masked_pkt_seq, masked_win_seq, pkt_masked_label, win_masked_label, self.y[item]
        else:
            return  self.x1[item], self.x2[item], self.y[item]

    def random_pkt_seq(self, pkt_seq, win_seq):
        seq_len = pkt_seq.shape[-1]
        masked_index = np.sort(np.random.choice(range(seq_len), 1, replace=False))
        pkt_masked_label = torch.ones_like(pkt_seq) * -1
        win_masked_label = torch.ones_like(win_seq) * -1
        pkt_masked_label[masked_index] = pkt_seq[masked_index]
        win_masked_label[masked_index] = win_seq[masked_index]

        for i in masked_index:
            prob = random.random()
            if prob < 0.8:
                pkt_seq[i] = self.MASK
                win_seq[i] = self.MASK
            elif prob < 0.9:
                pkt_seq[i] = random.randrange(2, self.vocab_size1)
                win_seq[i] = random.randrange(2, self.vocab_size2)
            else:
                pkt_seq[i] = pkt_seq[i]
                win_seq[i] = win_seq[i]

        return pkt_seq, win_seq, pkt_masked_label, win_masked_label


class IoTDataset(Dataset):
    def __init__(self, data, max_pkt_len):
        self.pkt = torch.from_numpy(data[:, 1:max_pkt_len+1])
        self.win = torch.from_numpy(data[:, max_pkt_len+1:2*max_pkt_len+1])
        self.port = torch.from_numpy(data[:,-3:-1])
        self.ciphersuit = torch.from_numpy(data[:,-1])
        self.y = torch.from_numpy(data[:, 0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.pkt[item],self.win[item],self.port[item],\
               self.ciphersuit[item], self.y[item]


def get_dataloader(data,batch_size,max_len):
    temp = []
    for i in data:
        dataloader = iter(DataLoader(IoTDataset(i,max_len), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0))
        temp.append(dataloader)
    return temp

def fixed_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
