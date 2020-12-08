import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, pickle, random

np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
import torch.optim as optim
from model import ERC_transfo_xl, ERC_xlnet
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from trainer import train_or_eval_model_for_transfo_xl
from dataset import IEMOCAPDataset_transfo_xl
from dataloader import get_IEMOCAP_loaders_transfo_xl, get_IEMOCAP_loaders_xlnet
from transformers import AdamW
import logging

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


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

    parser.add_argument('--model_save_dir', type =str, default='saved_models/')

    parser.add_argument('--basemodel', type=str, default='xlnet_dialog',
                        choices=['xlnet', 'transfo_xl', 'bert', 'xlnet_dialog'],
                        help='base model')

    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[3, 3, 3, 3],
                        help='number of heads:[n_local,n_global,n_speaker,n_listener], default=[3,3,3,3]')

    parser.add_argument('--max_sent_len', type=int, default=200,
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument('--mem_len', type=int, default=900, help='max memory length')
    parser.add_argument('--windowp', type=int, default=10, help='local attention window size')
    parser.add_argument('--attn_type', type=str, choices=['uni', 'bi'], default='bi')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='IEMOCAP', choices=['IEMOCAP', 'MELD','DailyDialog','EmoryNLP'], type=str,
                        help='dataset name, IEMOCAP or MELD or DailyDialog')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=4, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=2, metavar='E', help='number of epochs')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    args = parser.parse_args()
    print(args)
    seed_everything()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    if args.basemodel == 'transfo_xl':
        train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_transfo_xl(
            dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)
    elif args.basemodel in ['xlnet', 'xlnet_dialog']:
        train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_xlnet(
            dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)

    n_classes = len(label_vocab['itos'])

    print('building model..')
    if args.basemodel == 'transfo_xl':
        model = ERC_transfo_xl(args, n_classes)
    elif args.basemodel in ['xlnet', 'xlnet_dialog']:
        model = ERC_xlnet(args, n_classes, use_cls=True)



    if cuda:
        model.cuda()


    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters(), lr=args.lr)


    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    best_testacc = 0
    best_testfscore = 0
    best_vadacc = 0
    best_vadfscore = 0

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore = train_or_eval_model_for_transfo_xl(model, loss_function,
                                                                                       train_loader, e, cuda,
                                                                                       args, optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model_for_transfo_xl(model, loss_function,
                                                                                       valid_loader, e, cuda, args)
        test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_model_for_transfo_xl(model,
                                                                                                     loss_function,
                                                                                                     test_loader,
                                                                                                     e, cuda, args)
        all_fscore.append([valid_fscore, test_fscore, test_acc])

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)


        if best_vadfscore < valid_fscore:
            best_vadfscore, best_vadacc = valid_fscore, valid_acc
            # if valid_fscore>63.:
            torch.save(model.state_dict(), args.model_save_dir+ args.dataset_name + '_'+args.basemodel+'.pth')
            print('Best model saved')


        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                       test_acc,
                       test_fscore, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    all_fscore = sorted(all_fscore, key=lambda x: x[0], reverse=True)
    print('Valid F-Score:', all_fscore[0][0], 'Test F-Score :', all_fscore[0][1], 'Test Acc: ', all_fscore[0][2])
    # logger.info('Best F-Score {},'.format(all_fscore[0][1]))
    # logger.info('all_fscore {}!'.format(all_fscore))



