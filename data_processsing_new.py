import numpy as np
#np.set_printoptions(threshold=np.sys.maxsize)
import os
import re
import random
from collections import Counter
from functools import reduce
from Config import ciphersuit_dict
import torch
from torch.utils.data import Dataset, DataLoader
# from config import *


class DataManager(object):
    def __init__(self, dataset, baseclasses, fewclasses, max_pkt_len, num_training_large_sample, num_testing_sample):
        self.dataset = dataset
        self.baseclasses = baseclasses
        self.fewclasses = fewclasses
        self.max_pkt_len = max_pkt_len
        self.num_training_large_sample = num_training_large_sample
        #self.num_training_few_sample = num_training_few_sample
        self.num_testing_sample = num_testing_sample
        self.PAD = 0
        self.UNK = 1
        self.MASK = 2
        self.PKT_PAD_WORD = '<pkt_blank>'
        self.PKT_UNK_WORD = '<pkt_unk>'
        self.PKT_MASK_WORD = '<pkt_mask>'
        self.WIN_PAD_WORD = '<win_blank>'
        self.WIN_UNK_WORD = '<win_unk>'
        self.WIN_MASK_WORD = '<win_mask>'
        self.baseclass_data, self.fewclass_data = self.read_instances(self.dataset, self.baseclasses, self.fewclasses, self.max_pkt_len)
        self.Large_training_datas, self.Large_validing_datas, self.testing_datas = self.train_test_split()


    def read_instances(self, dataset, baseclasses, fewclasses, max_pkt_len):
        baseclass_data = []
        fewclass_data = []
        for key in baseclasses.keys():
            file_name1 = os.path.join(dataset["net"], "tcp_{cla}.txt".format(cla=key))
            tcp_data = self.read_instances_from_file(file_name1, baseclasses, max_pkt_len)
            file_name2 = os.path.join(dataset, "udp_{cla}.txt".format(cla=i))
            udp_data = self.read_instances_from_file(file_name2, max_pkt_len)
            random.shuffle(tcp_data)
            #random.shuffle(udp_data)
            baseclass_data.append(tcp_data)
        for key in fewclasses.keys():
            file_name1 = os.path.join(dataset["iot"], "tcp_{cla}.txt".format(cla=key))
            tcp_data = self.read_instances_from_file(file_name1, fewclasses, max_pkt_len)
            file_name2 = os.path.join(dataset, "udp_{cla}.txt".format(cla=i))
            udp_data = self.read_instances_from_file(file_name2, max_pkt_len)
            random.shuffle(tcp_data)
            #random.shuffle(udp_data)
            fewclass_data.append(tcp_data)
        return baseclass_data, fewclass_data



    def read_instances_from_file(self, inst_file, classes_dict, max_pkt_len):
        alldata_insts = []
        with open(inst_file) as f:
            # traverse stream start
            for sent in f:
                words = sent.strip().split('\t')
                win_inst = words[-2]
                port_inst = words[-3]
                ciphersuit_inst = words[-1]
                pkt_inst = words[:(len(words) - 3)]
                if ('win:' in win_inst):
                    if len(win_inst) > 4:
                        #win_seq = list(map(lambda x: int(x), list(re.split(' |:', win_inst)[1:(max_pkt_len+1)])))
                        win_seq = list(re.split(' |:', win_inst)[1:(max_pkt_len + 1)])
                        win_seq = win_seq + [self.WIN_PAD_WORD] * (max_pkt_len - len(win_seq))
                    else:
                        win_seq = [self.WIN_PAD_WORD] * max_pkt_len
                #win_seq = [self.WIN_PAD_WORD] * max_pkt_len
                if ('port:' in port_inst):
                    port = list(map(lambda x: int(x), list(re.split(' |:', port_inst)[1:])))
                if ('ciphersuits:' in ciphersuit_inst):
                    ciphersuit = list(set(list(re.split(' |:', ciphersuit_inst))[1:]))
                    ciphersuit = ciphersuit[0]
                    if(ciphersuit!=''):
                        ciphersuit = ciphersuit.split('TLS_',1)[1].strip(')')

                    ciphersuit = ciphersuit_dict[ciphersuit]

                if (len(pkt_inst) != 0):
                    label = [classes_dict[pkt_inst[0]]]
                    #pkt_seq = list(map(lambda x:int(x),pkt_inst[1:(max_pkt_len+1)]))
                    pkt_seq = pkt_inst[1:(max_pkt_len + 1)]
                    pkt_seq = pkt_seq + [self.PKT_PAD_WORD] * (max_pkt_len - len(pkt_seq))
                    #pkt_seq = [self.PKT_PAD_WORD]*max_pkt_len
                now_word_inst = label + [pkt_seq] + [win_seq] + [port] + [ciphersuit]
                #print(now_word_inst)
                alldata_insts.append(now_word_inst)

        print('[Info] Get {} instances from {}'.format(len(alldata_insts), inst_file))
        return alldata_insts


    def train_test_split(self):
        Base_validing_data = [i[:100] for i in self.baseclass_data]
        Base_training_data = [i[100: 100+self.num_training_large_sample] for i in self.baseclass_data]
        testing_data = [i[:self.num_testing_sample] for i in self.fewclass_data]
        return  Base_training_data, Base_validing_data, testing_data


    def build_vocab_idx(self):
        full_pkt_vocab = [pkt for data in self.Large_training_datas for sample in data for pkt in sample[1]]
        full_win_vacab = [win for data in self.Large_training_datas for sample in data for win in sample[2]]

        pkt_word2idx = {
            self.PKT_PAD_WORD:self.PAD,
            self.PKT_UNK_WORD:self.UNK,
            self.PKT_MASK_WORD:self.MASK
        }
        win_word2idx = {
            self.WIN_PAD_WORD: self.PAD,
            self.WIN_UNK_WORD: self.UNK,
            self.WIN_MASK_WORD: self.MASK
        }

        ignored_pkt_word_count = 0
        ignored_win_word_count = 0

        pkt_word_count = Counter(full_pkt_vocab)
        win_word_count = Counter(full_win_vacab)

        for (word, count) in pkt_word_count.most_common():
            if word not in pkt_word2idx:
                if len(pkt_word2idx) < 1500:
                    pkt_word2idx[word] = len(pkt_word2idx)
                else:
                    ignored_pkt_word_count += 1

        for (word, count) in win_word_count.most_common():
            if word not in win_word2idx:
                if len(win_word2idx) < 4000:
                    win_word2idx[word] = len(win_word2idx)
                else:
                    ignored_win_word_count += 1

        print('[Info] Trimmed pkt vocabulary size = {},'.format(len(pkt_word2idx)))
        print('[Info] Trimmed win vocabulary size = {},'.format(len(win_word2idx)))
        print("[Info] Ignored pkt word count = {}".format(ignored_pkt_word_count))
        print("[Info] Ignored win word count = {}".format(ignored_win_word_count))

        return pkt_word2idx, win_word2idx

    def convert_instance_to_idx_seq(self, datas, pkt_word2idx, win_word2idx):
        return_datas = []
        for data in datas:
            temp = []
            for sample in data:
                pkt_seq = [pkt_word2idx.get(w, self.UNK) for w in sample[1]]
                win_seq = [win_word2idx.get(w, self.UNK) for w in sample[2]]
                sample = [sample[0]] + pkt_seq + win_seq + [sample[-2]] + [sample[-1]]
                temp.append(sample)
            return_datas.append(temp)
        return return_datas


    def getdata(self):
        pkt_word2idx, win_word2idx = self.build_vocab_idx()
        self.Large_training_datas = \
            self.convert_instance_to_idx_seq(self.Large_training_datas, pkt_word2idx, win_word2idx)
        self.Large_validing_datas = \
            self.convert_instance_to_idx_seq(self.Large_validing_datas, pkt_word2idx, win_word2idx)
        self.testing_datas = \
            self.convert_instance_to_idx_seq(self.testing_datas, pkt_word2idx, win_word2idx)
        Bert_training_data, Bert_validing_data = self.get_BERTDATA()
        training_datas, validing_datas, testing_datas = self.get_FewShotDATA()

        data = {
            "BERTDataset":{
                "train": Bert_training_data,
                "valid": Bert_validing_data
            },
            "FewShotDataset":{
                'train': training_datas,
                'valid': validing_datas,
                'test': testing_datas,
            }
        }
        return data

    def get_BERTDATA(self):
        temp1 = np.empty((0, 2 * self.max_pkt_len + 1), dtype=np.int64)
        for i in self.Large_training_datas:
            bert_data = list(map(lambda x:x[0:-2], i))
            bert_data = np.array(bert_data, dtype=np.int64)
            temp1 = np.vstack([temp1, bert_data])
        np.random.shuffle(temp1)

        temp2 = np.empty((0, 2 * self.max_pkt_len + 1), dtype=np.int64)
        for i in self.Large_validing_datas:
            bert_data = list(map(lambda x: x[0:-2], i))
            bert_data = np.array(bert_data, dtype=np.int64)
            if bert_data.shape[0] == 0:
                continue
            temp2 = np.vstack([temp2, bert_data])

        np.random.shuffle(temp2)
        return temp1, temp2

    def get_FewShotDATA(self):
        for data in self.Large_training_datas:
            for index, item in enumerate(data):
                data[index] = reduce(lambda x,y:x.extend(y) or x, [i if isinstance(i,list) else [i] for i in item])

        for data in self.Large_validing_datas:
            for index, item in enumerate(data):
                data[index] = reduce(lambda x,y:x.extend(y) or x, [i if isinstance(i,list) else [i] for i in item])

        for data in self.testing_datas:
            for index, item in enumerate(data):
                data[index] = reduce(lambda x,y:x.extend(y) or x, [i if isinstance(i,list) else [i] for i in item])

        return list(map(lambda x:np.array(x, dtype=np.int64),self.Large_training_datas)), \
               list(map(lambda x:np.array(x, dtype=np.int64),self.Large_validing_datas)), \
               list(map(lambda x:np.array(x, dtype=np.int64),self.testing_datas)),




def fixed_seed(seed):
    np.random.seed(seed)
    random.seed(seed)



