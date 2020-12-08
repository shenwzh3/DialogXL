import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import BertModel, BertConfig, TransfoXLModel, XLNetModel
from modeling_xlnet import XLNetModel_dialog

# For methods and models related to DialogueGCN jump to line 516
class BertERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)
        # bert_incoder
        self.bert = BertModel.from_pretrained(args.bert_model_dir)
        in_dim =  args.bert_dim

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers- 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, content_ids, token_types):

        # the embeddings for bert
        text_feature = self.bert(content_ids, token_type_ids = token_types)[1] #(N , D)
        text_feature = self.dropout(text_feature)

        # pooling

        outputs = self.out_mlp(text_feature) #(N, D)

        return outputs

class ERC_transfo_xl(nn.Module):
    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.transfo_xl = TransfoXLModel.from_pretrained(args.home_dir + args.bert_model_dir)
        in_dim =  args.bert_dim

        pool_layers = [nn.Linear(in_dim, args.hidden_dim),nn.ReLU()]
        self.pool_fc = nn.Sequential(*pool_layers)

        # output mlp layers
        layers = []
        for _ in range(args.mlp_layers):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, content_ids, mems, content_mask ):
        '''

        :param content_ids: (B, L)
        :param mems:  tuple
        :param content_mask: (B, L)
        :return:
        '''
        # the embeddings for bert
        # print('content_ids', content_ids.size())
        # print('content_max', content_mask.size())
        text_feature, new_mems = self.transfo_xl(content_ids, mems)[:2] #(B, L, D), tuple
        # print('tex_feature', text_feature.size())
        text_feature = self.pool_fc(text_feature) #(B, L, D)
        text_out = torch.max(text_feature * content_mask.unsqueeze(2), dim = 1)[0] #(B, D)
        # print('text_out', text_out.size())

        # pooling
        text_out = self.dropout(text_out)
        outputs = self.out_mlp(text_out) #(N, C)
        # print('outputs', outputs.size())
        # exit()
        return outputs, new_mems


class ERC_xlnet(nn.Module):
    def __init__(self, args, num_class, use_cls = True):
        super().__init__()
        self.args = args
        # gcn layer
        self.use_cls = use_cls

        self.dropout = nn.Dropout(args.dropout)

        if args.basemodel == 'xlnet':
            self.xlnet = XLNetModel.from_pretrained(args.bert_model_dir)
        elif args.basemodel == 'xlnet_dialog':
            self.xlnet = XLNetModel_dialog.from_pretrained(args.bert_model_dir)
        self.xlnet.mem_len = args.mem_len
        self.xlnet.attn_type = args.attn_type
        in_dim =  args.bert_dim

        pool_layers = [nn.Linear(in_dim, args.hidden_dim),nn.ReLU()]
        self.pool_fc = nn.Sequential(*pool_layers)

        # output mlp layers
        layers = []
        for _ in range(args.mlp_layers):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, content_ids, mems, content_mask, content_lengths, speaker_ids, speaker_mask, window_mask ):
        '''

        :param content_ids: (B, L)
        :param mems:  tuple
        :param content_mask: (B, L)
        :param content_lengths: (B, )
        :param speaker_ids: (B, )
        :return:
        '''
        # the embeddings for bert
        # print('content_ids', content_ids.size())
        # print('content_max', content_mask.size())

        if self.args.basemodel == 'xlnet':
            text_feature, new_mems = self.xlnet(input_ids=content_ids, mems=mems)[:2]  # (B, L, D), tuple
            speaker_mask = None
            window_mask = None
        elif self.args.basemodel == 'xlnet_dialog':
            text_feature, new_mems, speaker_mask, window_mask = self.xlnet(input_ids=content_ids, mems=mems, content_lengths = content_lengths,
                                                speaker_ids = speaker_ids, content_mask = content_mask, speaker_mask = speaker_mask,
                                                window_mask = window_mask, windowp = self.args.windowp, num_heads = self.args.num_heads)[:4]  # (B, L, D), tuple
        # print('tex_feature', text_feature.size())
        if self.use_cls:
            text_out = self.pool_fc(text_feature[:,0,:]) # (B, D)
        else:
            text_feature = self.pool_fc(text_feature) #(B, L, D)
            text_out = torch.max(text_feature * content_mask.unsqueeze(2), dim = 1)[0] #(B, D)

        # pooling
        # pooling
        text_out = self.dropout(text_out)
        outputs = self.out_mlp(text_out) #(N, C)
        # print('outputs', outputs.size())
        # exit()
        return outputs, new_mems, speaker_mask, window_mask