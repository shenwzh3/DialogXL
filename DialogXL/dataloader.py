from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from  transformers import BertTokenizer, TransfoXLTokenizer, XLNetTokenizer

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)


def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('../data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('../data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec_dir = '../data/%s/person_vect.pkl' % (dataset_name)
    # if os.path.exists(person_vec_dir):
    #     print('Load person vec from ' + person_vec_dir)
    #     person_vec = pickle.load(open(person_vec_dir, 'rb'))
    # else:
    #     print('Creating personality vectors')
    #     person_vec = np.random.randn(len(speaker_vocab['itos']), 100)
    #     print('Saving personality vectors to' + person_vec_dir)
    #     with open(person_vec_dir,'wb') as f:
    #         pickle.dump(person_vec, f, -1)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec

def read_datas(dataset_name, batch_size):
    # training set
    with open('../data/%s/train_data.json' % (dataset_name), encoding='utf-8') as f:
        train_raw = json.load(f)
    train_raw = sorted(train_raw,key = lambda x:len(x))
    new_train_raw = []
    for i in range(0, len(train_raw), batch_size):
        new_train_raw.append(train_raw[i:i+batch_size])

    with open('../data/%s/dev_data.json' % (dataset_name), encoding='utf-8') as f:
        dev_raw = json.load(f)
    dev_raw = sorted(dev_raw,key = lambda x:len(x))
    new_dev_raw = []
    for i in range(0, len(dev_raw), batch_size):
        new_dev_raw.append(dev_raw[i:i+batch_size])

    with open('../data/%s/test_data.json' % (dataset_name), encoding='utf-8') as f:
        test_raw = json.load(f)
    test_raw = sorted(test_raw,key = lambda x:len(x))
    new_test_raw = []
    for i in range(0, len(test_raw), batch_size):
        new_test_raw.append(test_raw[i:i+batch_size])

    return new_train_raw, new_dev_raw, new_test_raw


def get_IEMOCAP_loaders_transfo_xl(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    tokenizer = TransfoXLTokenizer.from_pretrained(args.home_dir + args.bert_tokenizer_dir)
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    train_data, dev_data, test_data = read_datas(dataset_name, batch_size)
    print('building datasets..')
    trainsets = [IEMOCAPDataset_transfo_xl(d,  speaker_vocab, label_vocab, args, tokenizer) for d in train_data]
    devsets = [IEMOCAPDataset_transfo_xl(d, speaker_vocab, label_vocab, args, tokenizer) for d in dev_data]
    testsets = [IEMOCAPDataset_transfo_xl(d, speaker_vocab, label_vocab, args, tokenizer) for d in test_data]

    return trainsets, devsets, testsets, speaker_vocab, label_vocab, person_vec

def get_IEMOCAP_loaders_xlnet(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    tokenizer = XLNetTokenizer.from_pretrained( args.bert_tokenizer_dir)
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    train_data, dev_data, test_data = read_datas(dataset_name, batch_size)
    print('building datasets..')
    trainsets = [IEMOCAPDataset_xlnet(d,  speaker_vocab, label_vocab, args, tokenizer) for d in train_data]
    devsets = [IEMOCAPDataset_xlnet(d, speaker_vocab, label_vocab, args, tokenizer) for d in dev_data]
    testsets = [IEMOCAPDataset_xlnet(d, speaker_vocab, label_vocab, args, tokenizer) for d in test_data]

    return trainsets, devsets, testsets, speaker_vocab, label_vocab, person_vec

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dir', type=str, default='/data/SHENWZH/')
    parser.add_argument('--bert_model_dir', type=str, default='models/chinese_wwm_ext_pytorch/')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='models/chinese_wwm_ext_pytorch/vocab.txt')
    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--max_sent_len', type=int, default=200,
                        help='max content length for each text, if set to 0, then the max length has no constrain')

    args = parser.parse_args()
    print(args)
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec = get_IEMOCAP_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = args)
    for b in train_loader:
        for d in b:
            print(d.size())
            print(d)
        exit()
