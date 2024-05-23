from Module import Comparison_Net, FE_embedding
from dataset import get_dataloader
import argparse
import os
import torch.utils.data
from data_processsing_new import DataManager
from Config import Scenario_set
from BERT.bert import BERT
from train_few_shot import train
from test_few_shot import test
from valid_few_shot import valid
import torch.nn.functional as F
import torch
import random
import  numpy as np
import time
import copy 
np.set_printoptions(threshold=np.inf)





def fixed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', metavar='INPUT', type=str, default='TMC', choices=["TMC", "IMC", "CIC", "ALL"])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=25)
    parser.add_argument('--test_batch_size', type=int, default=25)
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
    parser.add_argument('--num_way', type=int, default=5)
    parser.add_argument('--num_shot', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_model', default='C_NETModel_New/')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # opt.cuda = opt.no_cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES


    os.makedirs(opt.save_model, exist_ok=True)
    fixed_seed(opt.seed)

    # ========= Processing Dataset =========#

    datamanager = DataManager(Scenario_set[opt.scenario]['dataset'], Scenario_set[opt.scenario]['net_class_dict'],
                              Scenario_set[opt.scenario]['iot_class'], opt.max_pkt_seq_len,
                              opt.num_training_sample, opt.num_testing_sample)

    data = datamanager.getdata()
    
    print(opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')
    print(device)

    print("Building BERT model")
    bert = BERT(vocab_size1=opt.pkt_vocab_size, vocab_size2=opt.win_vocab_size, max_len=opt.max_pkt_seq_len,
                hidden=opt.hidden, n_layers=opt.layers, attn_heads=opt.attn_heads,
                d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout)

    print("load parameters for BERT model")

    bert_model_name = "BERTModel_New/" + Scenario_set[opt.scenario]['dataset']['net'].split('/')[-1] + '_hidden_' + str(
        opt.hidden) + '_pkt_' + str(opt.max_pkt_seq_len) + '_sample_' + str(opt.num_training_sample) + '_seed_' + str(
        opt.seed) + '.chkpt'

    
    bert.load_state_dict(torch.load(bert_model_name)['BERT'])
    bert = bert.to(device)


    C_Net = Comparison_Net(opt.max_pkt_seq_len, opt.hidden).to(device)
    optim = torch.optim.Adam(C_Net.parameters(), lr=opt.lr)

    CNET_model_name = opt.save_model + Scenario_set[opt.scenario]['dataset']['net'].split('/')[-1] + \
                      '_hidden_' + str(opt.hidden) + '_pkt_' + str(opt.max_pkt_seq_len) +\
                      '_shot_'+ str(opt.num_shot) + '_sample_' + str(opt.num_training_sample) +'_seed_'+str(opt.seed)+'.pt'

    # CNET_model_name = opt.save_model + Scenario_set[opt.scenario]['dataset']['net'].split('/')[-1] + \
    #                   '_hidden_' + str(opt.hidden) + '_pkt_' + str(opt.max_pkt_seq_len) +\
    #                   '_shot_'+ str(opt.num_shot) +'_seed_'+str(opt.seed)+'.pt'


    dict_train = {i: len(data['FewShotDataset']['train'][i]) for i in range(len(Scenario_set[opt.scenario]['net_class_dict']))}
    dict_valid = {i: len(data['FewShotDataset']['valid'][i]) for i in range(len(Scenario_set[opt.scenario]['net_class_dict']))}




    if opt.train:
        print("-----------------------TRAIN--------------------------")
        MAX_ACC = 0
        for epoch in range(opt.epochs):
            start = time.time()
            train_dataloader = get_dataloader(data['FewShotDataset']['train'], batch_size=opt.train_batch_size,
                                            max_len=opt.max_pkt_seq_len)
            dict_train_copy = copy.deepcopy(dict_train)
            task, total_loss, train_acc = train(bert, C_Net, epoch, train_dataloader, optim, dict_train_copy, 
                device, opt.num_way, opt.num_shot, opt.train_batch_size)
            print(f"  - (Training)   epoch:{epoch + 1}, task:{task}, ave_loss:{total_loss/task}, acc:{train_acc}, elapse:{(time.time()-start)/60:.3f} min")

            start = time.time()
            dict_valid_copy = copy.deepcopy(dict_valid)
            Valid_acc = valid(bert, C_Net, data['FewShotDataset']['valid'], dict_valid_copy, device, len(Scenario_set[opt.scenario]['net_class_dict']), 
                opt.num_shot, opt.test_batch_size, opt.max_pkt_seq_len)
            print(f"  - (Validing)   epoch:{epoch + 1}, acc:{Valid_acc}, elapse:{(time.time()-start)/60:.3f} min")
            
            if Valid_acc >= MAX_ACC:
                torch.save(C_Net.state_dict(), CNET_model_name)
                MAX_ACC = Valid_acc
                print("  - update model!")



    C_Net.load_state_dict(torch.load(CNET_model_name))
    print("-----------------------TEST--------------------------")

    dict_test = {i: len(data['FewShotDataset']['test'][i]) for i in range(len(Scenario_set[opt.scenario]['iot_class']))}
    

    test(bert, C_Net, data['FewShotDataset']['test'], dict_test, device, len(Scenario_set[opt.scenario]['iot_class']),
         opt.num_shot, opt.test_batch_size, opt.max_pkt_seq_len)
