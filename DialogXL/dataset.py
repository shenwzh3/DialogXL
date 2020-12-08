import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import argparse
import random
from  transformers import BertTokenizer, TransfoXLTokenizer, XLNetTokenizer

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


class IEMOCAPDataset_transfo_xl(Dataset):

    def __init__(self, data, speaker_vocab=None, label_vocab=None, args = None, tokenizer = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.content_ids, self.labels, self.content_mask, self.ids = self.process(data, tokenizer)

        self.len = len(self.content_ids)

    def process(self, data, tokenizer):
        '''
        :param data:
            list of dialogues
        :param tokenizer:
        :return:
        '''
        max_dialog_len = max([len(d) for d in data])
        content_ids = [] # list, content_ids for each segment
        labels = [] # list, labels for each segment
        content_mask = [] # list, masks of tokens for each segment
        indecies = []

        for i in range(max_dialog_len):
            ids = []
            lbs = []
            mask = []
            for d in data:
                if i < len(d):
                    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d[i]['text'], add_space_before_punct_symbol=True))
                    m = [1 for i in range(len(token_ids))]
                    ids.append(token_ids)
                    mask.append(m)
                    lbs.append(self.label_vocab['stoi'][d[i]['label']] if 'label' in d[i].keys() else -1)
                else:
                    ids.append([0])
                    mask.append([0])
                    lbs.append(-1)
            content_ids.append(ids)
            labels.append(lbs)
            content_mask.append(mask)
            indecies.append(i)

        for i in range(max_dialog_len):
            max_sent_len = min(max([len(x) for x in content_ids[i]]), self.args.max_sent_len)
            if max_sent_len == 0:
                max_sent_len = 1
            for j in range(len(content_ids[i])):
                if len(content_ids[i][j]) > max_sent_len:
                    content_ids[i][j] = content_ids[i][j][:max_sent_len]
                    content_mask[i][j] = content_mask[i][j][:max_sent_len]
                else:
                    content_ids[i][j] += [0] * (max_sent_len - len(content_ids[i][j]))
                    content_mask[i][j] += [0] * (max_sent_len - len(content_mask[i][j]))
                # print(len(content_mask[i][j]))
            # print('\n')

        # exit()

        return content_ids, labels, content_mask, indecies

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        return torch.LongTensor(self.content_ids[index]), \
               torch.LongTensor(self.labels[index]), \
               torch.FloatTensor(self.content_mask[index]), \
               self.ids[index]

    def __len__(self):
        return self.len

    # def collate_fn(self, data):
    #     '''
    #
    #     :param data:
    #         content_ids
    #         token_types
    #         labels
    #     :return:
    #
    #     '''
    #     content_ids = pad_sequence([d[0] for d in data], batch_first = True) # (B, T, )
    #     token_types = pad_sequence([d[1] for d in data], batch_first = True) # (B, T, )
    #     labels = torch.LongTensor([d[2] for d in data])
    #
    #     return content_ids, token_types, labels


class IEMOCAPDataset_xlnet(Dataset):

    def __init__(self, data, speaker_vocab=None, label_vocab=None, args = None, tokenizer = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.content_ids, self.labels, self.content_mask, self.content_lengths, self.speaker_ids, self.ids = self.process(data, tokenizer)

        self.len = len(self.content_ids)

    def process(self, data, tokenizer):
        '''
        :param data:
            list of dialogues
        :param tokenizer:
        :return:
        '''
        max_dialog_len = max([len(d) for d in data])
        content_ids = [] # list, content_ids for each segment
        labels = [] # list, labels for each segment
        content_mask = [] # list, masks of tokens for each segment
        content_lengths = [] # list, length for each utterace in each segment
        speaker_ids = []
        indecies = []

        for i in range(max_dialog_len):
            ids = []
            lbs = []
            mask = []
            speaker = []
            for d in data:
                if i < len(d):
                    token_ids = tokenizer.convert_tokens_to_ids(['<cls>'] + tokenizer.tokenize(d[i]['text']))
                    m = [1 for i in range(len(token_ids))]
                    ids.append(token_ids)
                    mask.append(m)
                    speaker.append(self.speaker_vocab['stoi'][d[i]['speaker']] + 1)
                    lbs.append(self.label_vocab['stoi'][d[i]['label']] if 'label' in d[i].keys() else -1)
                else:
                    ids.append([3])
                    mask.append([1])
                    lbs.append(-1)
                    speaker.append(0)
            content_ids.append(ids)
            labels.append(lbs)
            content_mask.append(mask)
            speaker_ids.append(speaker)
            indecies.append(i)

        for i in range(max_dialog_len):
            lens = []
            max_sent_len = min(max([len(x) for x in content_ids[i]]), self.args.max_sent_len)
            if max_sent_len == 0:
                max_sent_len = 1
            for j in range(len(content_ids[i])):
                if len(content_ids[i][j]) > max_sent_len:
                    lens.append(max_sent_len)
                    content_ids[i][j] = content_ids[i][j][:max_sent_len]
                    content_mask[i][j] = content_mask[i][j][:max_sent_len]
                else:
                    lens.append(len(content_ids[i][j]))
                    content_ids[i][j] += [5] * (max_sent_len - len(content_ids[i][j]))
                    content_mask[i][j] += [0] * (max_sent_len - len(content_mask[i][j]))
            content_lengths.append(lens)
                # print(len(content_mask[i][j]))
            # print('\n')

        # exit()

        return content_ids, labels, content_mask, content_lengths, speaker_ids, indecies

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        return torch.LongTensor(self.content_ids[index]), \
               torch.LongTensor(self.labels[index]), \
               torch.FloatTensor(self.content_mask[index]), \
               torch.LongTensor(self.content_lengths[index]),\
               torch.FloatTensor(self.speaker_ids[index]),\
               self.ids[index]

    def __len__(self):
        return self.len

if __name__ == "__main__":
    import json
    # data = json.load(open('/data1/SHENWZH/quadruples/data/2020_5_13/train_data.json', encoding='utf-8'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dir', type=str, default='/data/SHENWZH/')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='models/xlnet_base/')
    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--max_sent_len', type=int, default=128,
                        help='max content length for each text, if set to 0, then the max length has no constrain')

    args = parser.parse_args()
    speaker_vocab, label_vocab, person_vec = load_vocab('IEMOCAP')
    tokenizer = XLNetTokenizer.from_pretrained(args.home_dir + args.bert_tokenizer_dir)
    with open('../data/IEMOCAP/train_data.json.feature' , encoding='utf-8') as f:
        train_raw = json.load(f)
    train_raw = sorted(train_raw,key = lambda x:len(x))
    train_set = IEMOCAPDataset_xlnet(train_raw[:32], speaker_vocab, label_vocab, args, tokenizer)
    # for d in train_set:
    # #     print(d[0])
    #     print(d[0].size())
    # #     # print(d[1])
    # #     # print(d[2].size())
    # #     # print(d[3])


