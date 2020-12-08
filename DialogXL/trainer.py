import numpy as np, argparse, time, pickle, random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from utils import person_embed
from tqdm import tqdm


def train_or_eval_model_for_transfo_xl(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, train=False):
    '''

    :param model:
    :param loss_function:
    :param dataloader: list of datasets,
    :param args:
    :param optimizer:
    :param train:
    :return:
    '''
    losses, preds, labels = [], [], []
    scores, vids = [], []


    assert not train or optimizer != None
    if train:
        model.train()
        # dataloader = tqdm(dataloader)
    else:
        model.eval()

    for dataset in dataloader:
        mems = None
        speaker_mask = None
        window_mask = None
        # if train:
        #     dataset = tqdm(dataset)
        # cnt = 0
        for data in dataset:
            if train:
                optimizer.zero_grad()

            # cnt += 1
            # if cnt > 4:
            #     exit()

            # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
            content_ids, label, content_mask, content_lengths, speaker_ids,_ = data
            # print(content_ids.size())
            # print(label.size())
            # print(content_mask.size())
            # print(content_lengths.size())
            # print(content_ids.size())
            # print(speaker_ids)
            # speaker_vec = person_embed(speaker_ids, person_vec)
            if cuda:
                content_ids = content_ids.cuda()
                content_mask = content_mask.cuda()
                speaker_ids = speaker_ids.cuda()
                content_lengths = content_lengths.cuda()
                label = label.cuda()

            if args.basemodel == 'transfo_xl':
                logits, mems = model(content_ids, mems, content_mask)
            elif args.basemodel in ['xlnet_dialog', 'xlnet']:
                logits, mems, speaker_mask, window_mask = model(content_ids = content_ids, mems = mems, content_mask = content_mask,
                                                   content_lengths = content_lengths, speaker_ids = speaker_ids,
                                                   speaker_mask = speaker_mask, window_mask = window_mask)

            loss = loss_function(logits, label)
            # print(speaker_mask.detach().cpu().numpy())
            # print(content_lengths)
            # print(speaker_ids)
            # print('------------------------------------------------------')


            label = label.cpu().numpy().tolist()
            pred = torch.argmax(logits, 1).cpu().numpy().tolist()
            # print(label)
            # print(pred)
            for l,p in zip(label, pred):
                if l != -1:
                    preds.append(p)
                    labels.append(l)
            losses.append(loss.item())
            # print(content_lengths)
            # print(mems[0].size())

            if train:
                loss_val = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.tensorboard:
                    for param in model.named_parameters():
                        writer.add_histogram(param[0], param[1].grad, epoch)
                optimizer.step()
            # torch.cuda.empty_cache()

    if preds != []:
        preds = np.array(preds)
        labels = np.array(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    # print(preds.tolist())
    # print(labels.tolist())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    else:
        avg_fscore = round(f1_score(labels, preds, average='micro', labels = list(range(1,7))) * 100, 2)
    # del mems
    # print(list(preds))
    # print(list(labels))

    return avg_loss, avg_accuracy, labels, preds, avg_fscore
