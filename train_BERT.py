import argparse
import numpy as np
import math
import time
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from data_processsing_new import DataManager
from dataset import BERTDataset
from Config import Scenario_set
from BERT.bert import BERT
from BERT.language_model import BERTLM
from BERT.Bert_optim_schedule import ScheduledOptim

train_loss_list = []
valid_loss_list = []


def fixed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''
    model.train()
    total_pkt_correct = 0
    total_win_correct = 0
    total_element = 0
    total_loss = 0
    count = 0
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        masked_pkt_seq, masked_win_seq, pkt_masked_label, win_masked_label, _ = map(lambda x: x.to(device), batch)

        loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean').to(device)

        pkt_pred, win_pred = model(masked_pkt_seq, masked_win_seq)
        pkt_pred = pkt_pred.transpose(1, 2)
        win_pred = win_pred.transpose(1, 2)
        optimizer.zero_grad()
        loss = loss_fn(pkt_pred, pkt_masked_label) + 0.5 * loss_fn(win_pred, win_masked_label)
        loss.backward()
        optimizer.step()
        count += 1

        y_pkt_predict = torch.max(pkt_pred, 1)[1][pkt_masked_label != -1]
        y_pkt_true = pkt_masked_label[pkt_masked_label != -1]
        y_win_predict = torch.max(win_pred, 1)[1][win_masked_label != -1]
        y_win_true = win_masked_label[win_masked_label != -1]

        pkt_correct = y_pkt_predict.eq(y_pkt_true).sum().item()
        win_correct = y_win_predict.eq(y_win_true).sum().item()
        total_pkt_correct += pkt_correct
        total_win_correct += win_correct
        total_element += y_win_true.shape[-1]
        total_loss += loss.item()


    pkt_valid_acc = total_pkt_correct/total_element * 100.
    win_valid_acc = total_win_correct/total_element * 100.
    return total_loss / count, pkt_valid_acc, win_valid_acc


def valid_epoch(model, validing_data, device):
    ''' Epoch operation in valid phase'''
    model.eval()
    total_pkt_correct = 0
    total_win_correct = 0
    total_element = 0
    total_loss = 0
    count = 0
    for batch in tqdm(
            validing_data, mininterval=2,
            desc='  - (Validing)   ', leave=False):
        masked_pkt_seq, masked_win_seq, pkt_masked_label, win_masked_label, _ = map(lambda x: x.to(device), batch)

        loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean').to(device)

        pkt_pred, win_pred = model(masked_pkt_seq, masked_win_seq)
        pkt_pred = pkt_pred.transpose(1, 2)
        win_pred = win_pred.transpose(1, 2)

        loss = loss_fn(pkt_pred, pkt_masked_label) + 0.5 * loss_fn(win_pred, win_masked_label)
        count += 1

        y_pkt_predict = torch.max(pkt_pred, 1)[1][pkt_masked_label != -1]
        y_pkt_true = pkt_masked_label[pkt_masked_label != -1]
        y_win_predict = torch.max(win_pred, 1)[1][win_masked_label != -1]
        y_win_true = win_masked_label[win_masked_label != -1]

        pkt_correct = y_pkt_predict.eq(y_pkt_true).sum().item()
        win_correct = y_win_predict.eq(y_win_true).sum().item()
        total_pkt_correct += pkt_correct
        total_win_correct += win_correct
        total_element += y_win_true.shape[-1]
        total_loss += loss.item()


    pkt_valid_acc = total_pkt_correct/total_element * 100.
    win_valid_acc = total_win_correct/total_element * 100.
    return total_loss / count, pkt_valid_acc, win_valid_acc



def train(model, training_data, validing_data, optimizer, device, opt):

    # valid_accus = []
    # test_accus=[]
    MAX_VALID_ACC = [0,0]
    for epoch_i in range(opt.epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_pkt_accu, train_win_accu = train_epoch(model, training_data, optimizer, device)
        train_loss_list.append(train_loss)

        print('  - (Training)   ppl: {ppl: 8.5f}, pkt_accuracy: {accu1:3.3f} %, win_accuracy: {accu2:3.3f} %,'\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu1=train_pkt_accu, accu2=train_win_accu,
                  elapse=(time.time()-start)/60))

        #
        start = time.time()
        valid_loss, valid_pkt_accu, valid_win_accu = valid_epoch(model, validing_data , device)
        valid_loss_list.append(valid_loss)
        print('  - (Validation) ppl: {ppl: 8.5f}, pkt_accuracy: {accu1:3.3f} %, win_accuracy: {accu2:3.3f} %,'\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu1=valid_pkt_accu, accu2=valid_win_accu,
                    elapse=(time.time()-start)/60))
        
        # valid_accus += [valid_accu]

        model_state_dict1 = model.bert.state_dict()
        model_state_dict2 = [model.mask_lm1.state_dict(), model.mask_lm2.state_dict()]
        checkpoint = {
            'BERT': model_state_dict1,
            'MASK_LM': model_state_dict2,
            'settings': opt,
            'epoch': epoch_i}

        model_name = opt.save_model + Scenario_set[opt.scenario]['dataset']['net'].split('/')[-1] + \
                         '_hidden_'+ str(opt.hidden) + '_pkt_'+ str(opt.max_pkt_seq_len) + '_sample_' + str(opt.num_training_sample) + '_seed_'+str(opt.seed) + '.chkpt'
        
        # if opt.save_mode == 'best':
        #     print(MAX_VALID_ACC)
        #     if valid_pkt_accu >= MAX_VALID_ACC[0] and valid_win_accu >= MAX_VALID_ACC[1]:
        #         torch.save(checkpoint, model_name)
        #         MAX_VALID_ACC = [valid_pkt_accu, valid_win_accu]
        #         print("update model!")
        # elif opt.save_mode == 'all':
        #     torch.save(checkpoint, model_name)
        # else:
        #     pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', metavar='INPUT', type=str, default='TMC', choices=["TMC","IMC","CIC"])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_training_sample', type=int, default=5000)
    parser.add_argument('--num_validing_sample', type=int, default=30)
    parser.add_argument('--num_testing_sample', type=int, default=100)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)

    parser.add_argument('--attn_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)

    parser.add_argument('--pkt_vocab_size', type=int, default=1500)
    parser.add_argument('--win_vocab_size', type=int, default=4000)
    parser.add_argument('--max_pkt_seq_len', type=int, default=16)
    #parser.add_argument('--masked_ratio', type=float, default=0.125)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--betas', type=tuple, default=(0.9,0.999))
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=10000)


    # parser.add_argument('-brn', type=int, default=5)
    # parser.add_argument('-log', default=None)

    parser.add_argument('--save_model', default='BERTModel_New/')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES

    os.makedirs(opt.save_model, exist_ok=True)

    fixed_seed(opt.seed)
    # ========= Processing Dataset =========#

    datamanager = DataManager(Scenario_set[opt.scenario]['dataset'], Scenario_set[opt.scenario]['net_class_dict'],
                              Scenario_set[opt.scenario]['iot_class'], opt.max_pkt_seq_len,
                              opt.num_training_sample, opt.num_testing_sample)

    data = datamanager.getdata()
    training_data, validing_data = prepare_dataloaders(data, opt)
    print(opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')
    print(device)

    print("Building BERT model")
    bert = BERT(vocab_size1=opt.pkt_vocab_size, vocab_size2=opt.win_vocab_size,max_len= opt.max_pkt_seq_len,
                hidden=opt.hidden, n_layers=opt.layers, attn_heads=opt.attn_heads,
                d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout)

    bertlm = BERTLM(bert, opt.pkt_vocab_size, opt.win_vocab_size).to(device)

    optim = Adam(bertlm.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=opt.weight_decay)
    optim_schedule = ScheduledOptim(optim, bert.hidden, n_warmup_steps=opt.warmup_steps)

    train(bertlm, training_data, validing_data, optim, device, opt)

    print(train_loss_list)
    print(valid_loss_list)

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


    valid_loader = torch.utils.data.DataLoader(
        BERTDataset(
            data=data["BERTDataset"]["valid"],
            vocab_size1 = opt.pkt_vocab_size,
            vocab_size2 = opt.win_vocab_size
        ),
        num_workers=4,
        batch_size=opt.eval_batch_size,
        shuffle=True)

    return train_loader, valid_loader


if __name__ == "__main__":
    main()
