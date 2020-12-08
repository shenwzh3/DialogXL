import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import numpy as np, argparse, time, pickle, random
np.set_printoptions(threshold = np.inf)
import torch
import torch.nn as nn
import torch.optim as optim
from model import ERC_transfo_xl, ERC_xlnet
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from trainer import  train_or_eval_model_for_transfo_xl
from dataset import IEMOCAPDataset_transfo_xl
from dataloader import get_IEMOCAP_loaders_transfo_xl, get_IEMOCAP_loaders_xlnet
from transformers import AdamW
import logging
# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100



def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_dir', type=str, default='/data/SHENWZH/models/xlnet_base/')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='/data/SHENWZH/models/xlnet_base/')
    parser.add_argument('--basemodel', type=str, default='xlnet_dialog', choices=['xlnet', 'transfo_xl', 'bert', 'xlnet_dialog'],
                        help = 'base model')

    parser.add_argument('--dataset_name', default='IEMOCAP',  type= str, help='dataset name, IEMOCAP or MELD or DailyDialog')

    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    parser.add_argument('--num_heads', nargs='+', type=int, default=[3,3,3,3],
                        help='number of heads:[n_local,n_global,n_speaker,n_listener], default=[3,3,3,3]')

    parser.add_argument('--max_sent_len', type=int, default=200,
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument('--mem_len', type=int, default=1000, help='max memory length')
    parser.add_argument('--windowp', type=int, default=10, help='local attention window size')
    parser.add_argument('--attn_type', type = str, choices=['uni','bi'], default='bi')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dropout', type=float, default=0, metavar='dropout', help='dropout rate')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')

    parser.add_argument('--modelname', type=str, default='IEMOCAP_xlnet_dialog',
                        help='model name')

    parser.add_argument('--modelpath', type=str, default='saved_models/', help='model path')



    args = parser.parse_args()
    print(args)
    seed_everything()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    
    cuda = args.cuda
    batch_size = args.batch_size
    modelPath=args.modelpath + args.modelname +'.pth'
    if args.basemodel == 'transfo_xl':
        train_loader, valid_loader, test_loader,speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_transfo_xl(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
    elif args.basemodel in ['xlnet', 'xlnet_dialog']:
        train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_xlnet(
            dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)

    n_classes = len(label_vocab['itos'])

    print('building model..')
    if args.basemodel == 'transfo_xl':
        model = ERC_transfo_xl(args, n_classes)
    elif args.basemodel in ['xlnet','xlnet_dialog']:
        model = ERC_xlnet(args, n_classes, use_cls = True)
    if cuda:
        model.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters() , lr=args.lr)


    # loading model...
    print('Loading model from: {}'.format(modelPath))
    model.load_state_dict(torch.load(modelPath))

    print('Begin evaluation')
    valid_loss, valid_acc, _, _, valid_fscore= train_or_eval_model_for_transfo_xl(model, loss_function,
                                                                                            valid_loader, 1, cuda, args)
    test_loss, test_acc, test_label, test_pred, test_fscore= train_or_eval_model_for_transfo_xl(model,
                                                                                                        loss_function,
                                                                                                        test_loader,
                                                                                                        1, cuda, args)
    print('valid_acc: {}, valid_fscore: {}'.format(valid_acc,valid_fscore))
    print('test_acc: {}, test_fscore: {}'.format(test_acc,test_fscore))





