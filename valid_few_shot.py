import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(sci_mode=False)

import copy
from sklearn.metrics import balanced_accuracy_score
from dataset import get_dataloader
import torch.utils.data
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix


def valid(model1, model2, data, dict1, device, num_way ,num_shot, batch_size, max_len, ):
    model1.eval()
    model2.eval()

    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)

    num_task = 0
    index2 = list(range(num_way * (batch_size - num_shot)))

    with torch.no_grad():
        while num_task < 100:
            dataloader = get_dataloader(data, batch_size, max_len)
            samples_pkt = torch.LongTensor(0)
            samples_win = torch.LongTensor(0)
            samples_port = torch.LongTensor(0)
            samples_ciphersuit = torch.LongTensor(0)
            samples_label = torch.LongTensor(0)

            querys_pkt = torch.LongTensor(0)
            querys_win = torch.LongTensor(0)
            querys_port = torch.LongTensor(0)
            querys_ciphersuit = torch.LongTensor(0)
            querys_label = torch.LongTensor(0)

            way = random.sample(list(dict1.keys()), num_way)
            for i in way:
                pkt, win, port, ciphersuit, label = next(dataloader[i])
                samples_pkt = torch.cat((samples_pkt, pkt[:num_shot]), 0)
                samples_win = torch.cat((samples_win, win[:num_shot]), 0)
                samples_label = torch.cat((samples_label, label[:num_shot]), 0)
                samples_port = torch.cat((samples_port, port[:num_shot]), 0)
                samples_ciphersuit = torch.cat((samples_ciphersuit, ciphersuit[:num_shot]), 0)

                querys_pkt = torch.cat((querys_pkt, pkt[num_shot:]), 0)
                querys_win = torch.cat((querys_win, win[num_shot:]), 0)
                querys_label = torch.cat((querys_label, label[num_shot:]), 0)
                querys_port = torch.cat((querys_port, port[num_shot:]), 0)
                querys_ciphersuit = torch.cat((querys_ciphersuit, ciphersuit[num_shot:]), 0)

            random.shuffle(index2)
            samples_pkt = samples_pkt.to(device)
            samples_win = samples_win.to(device)
            samples_port = samples_port.to(device)
            samples_ciphersuit = samples_ciphersuit.to(device)
            samples_label = samples_label.to(device)

            querys_pkt = querys_pkt[index2].to(device)
            querys_win = querys_win[index2].to(device)
            querys_label = querys_label[index2].to(device)
            querys_port = querys_port[index2].to(device)
            querys_ciphersuit = querys_ciphersuit[index2].to(device)

            samples_feature = model1(samples_pkt, samples_win)
            querys_feature = model1(querys_pkt, querys_win)
            

            pre = torch.LongTensor(0).to(device)

            for i in range(querys_feature.shape[0]):
                one_of_querys_feature = querys_feature[i].unsqueeze(0).repeat(samples_feature.shape[0],1,1)
                srcport_score = torch.where(samples_port - querys_port[i][0] == 0, 1, 0)
                dstport_score = torch.where(samples_port - querys_port[i][1] == 0, 1, 0)
                port_score = torch.sum(srcport_score + dstport_score, 1)
                port_score = torch.where(port_score - torch.ones_like(port_score) >= 0,1,0).unsqueeze(1).float()
                ciphersuit_score = torch.where(samples_ciphersuit - querys_ciphersuit[i] == 0, 1, 0).unsqueeze(1).float()
                Delta_Score = model2(samples_feature, one_of_querys_feature) 
                pre = torch.cat((pre, torch.argmax(Delta_Score,0)),0)


            pred = samples_label[pre]

            y_predict = torch.cat([y_predict, pred], 0)
            y_true = torch.cat([y_true, querys_label], 0)
            num_task += 1


    y_true_trans = np.array(y_true.cpu().numpy().tolist())
    y_predict_trans = np.array(y_predict.cpu().numpy().tolist())
    acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
    acc = 100. * acc
    return acc


if __name__ == '__main__':
    pass
