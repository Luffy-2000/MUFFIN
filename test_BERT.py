import argparse
import numpy as np
import math
import time
import os


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from  data_processing import DataManager
from dataset import BERTDataset
from Config import Scenario_set
from BERT.bert import BERT
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from BERT.language_model import BERTLM
from BERT.Bert_optim_schedule import ScheduledOptim



def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''
    model.train()

    total_correct = 0
    total_element = 0
    total_loss = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        masked_pkt_seq, masked_label, _ = map(lambda x: x.to(device), batch)

        loss_fn = nn.NLLLoss(ignore_index=-1).to(device)

        pred = model(masked_pkt_seq).transpose(1, 2)

        # forward
        optimizer.zero_grad()
        loss = loss_fn(pred, masked_label)
        loss.backward()
        optimizer.step_and_update_lr()


        y_predict = torch.max(pred, 1)[1][masked_label != -1]
        y_true = masked_label[masked_label != -1]

        correct = y_predict.eq(y_true).sum().item()
        total_correct += correct
        total_element += y_true.shape[-1]
        total_loss += loss.item()


    valid_acc = total_correct/total_element * 100.
    print("Training acc is: ", valid_acc, "Training loss is: ", total_loss)
    return total_loss, valid_acc


def test(model, testing_data, device):
    model.eval()
    features = []
    lables = []
    with torch.no_grad():
        for batch in tqdm(
                testing_data, mininterval=2,
                desc='  - (Testing) ', leave=False):
            pkt_seq, win_seq, label= map(lambda x: x.to(device), batch)

            feature = model(pkt_seq, win_seq)
            features.append(torch.mean(feature,-1))
            lables.append(label)
    all_features = torch.cat(features).tolist()
    all_label = torch.cat(lables).tolist()
    DrawTSNE(all_features,all_label)
 
def DrawTSNE(features, label):
    print("Starting compute t-SNE Embedding...")

    ts = TSNE(n_components=2, init="pca", random_state=0)
    # t-SNE降维
    reslut = ts.fit_transform(features)
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, label, "t-SNE Embedding of digits")
    # 显示图像
    plt.show()

def DrawPCA(features, label):
    print("Starting compute PCA Embedding...")
    pca = PCA(n_components=2)
    result = pca.fit_transform(features)
    fig = plot_embedding(result, label, "PCA Embedding of digits")
    plt.show()


def plot_embedding(data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
        fig = plt.figure()  # 创建图形实例
        ax = plt.subplot(111)  # 创建子图
        # 遍历所有样本
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set2(label[i]),
                     fontdict={"family":'Times New Roman', "size": 14})
        plt.xticks()  # 指定坐标的刻度
        plt.yticks()
        plt.title(title, fontsize=14)

        return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', metavar='INPUT', type=str, default='TMC', choices=["TMC","IMC"])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_training_sample', type=int, default=2000)
    parser.add_argument('--num_validing_sample', type=int, default=30)
    parser.add_argument('--num_testing_sample', type=int, default=100)


    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=2)

    parser.add_argument('--attn_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)

    #parser.add_argument('--vocab_size', type=int, default=1501)
    parser.add_argument('--pkt_vocab_size', type=int, default=1501)
    parser.add_argument('--win_vocab_size', type=int, default=65535)
    parser.add_argument('--max_pkt_seq_len', type=int, default=8)
    parser.add_argument('--masked_ratio', type=float, default=0.125)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--betas', type=tuple, default=(0.9,0.999))
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('-save_model', default='BERTModel/')
    parser.add_argument('-CUDA_VISIBLE_DEVICES', type=str, default='0')


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # opt.cuda = opt.no_cuda

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES

    os.makedirs(opt.save_model, exist_ok=True)

    # ========= Processing Dataset =========#

    datamanager = DataManager(Scenario_set[opt.scenario]['dataset'], Scenario_set[opt.scenario]['base_class'],
                              Scenario_set[opt.scenario]['all_class'], opt.max_pkt_seq_len,
                              opt.num_training_sample, opt.num_validing_sample, opt.num_testing_sample)

    data = datamanager.getdata()
    training_data,  testing_data = prepare_dataloaders(data, opt)
    print(opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')
    print(device)

    print("Building BERT model")
    bert = BERT(vocab_size1=opt.pkt_vocab_size, vocab_size2=opt.win_vocab_size, max_len= opt.max_pkt_seq_len,
                hidden=opt.hidden, n_layers=opt.layers, attn_heads=opt.attn_heads,
                d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout)

    print("load parameters for BERT model")

    model_name = opt.save_model + Scenario_set[opt.scenario]['dataset'].split('/')[-1] + \
                         '_hidden_'+ str(opt.hidden) + '_pkt_'+ str(opt.max_pkt_seq_len) + '.chkpt'
    bert.load_state_dict(torch.load(model_name)['BERT'])
    bert = bert.to(device)

    test(bert, testing_data, device)





def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        BERTDataset(
            data = data["BERTDataset"]["train"],
            vocab_size1 = opt.pkt_vocab_size,
            vocab_size2 = opt.win_vocab_size
        ),
        num_workers=4,
        batch_size=opt.batch_size,
        shuffle=True)


    test_loader = torch.utils.data.DataLoader(
        BERTDataset(
            data=data["BERTDataset"]["test"],
            vocab_size1=opt.pkt_vocab_size,
            vocab_size2=opt.win_vocab_size,
            masked=False
        ),
        num_workers=4,
        batch_size=opt.eval_batch_size,
        shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    main()
