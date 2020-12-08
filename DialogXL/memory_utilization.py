import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, pickle, random
np.set_printoptions(threshold = np.inf)
from tqdm import tqdm
import torch
from dataloader import get_IEMOCAP_loaders_transfo_xl, get_IEMOCAP_loaders_xlnet

seed = 100
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--home_dir', type=str, default='/data/SHENWZH/')
parser.add_argument('--bert_model_dir', type=str, default='models/xlnet_base/')
parser.add_argument('--bert_tokenizer_dir', type=str, default='models/xlnet_base/')

parser.add_argument('--basemodel', type=str, default='xlnet_dialog', choices=['xlnet', 'transfo_xl', 'bert', 'xlnet_dialog'],
                    help = 'base model')

parser.add_argument('--bert_dim', type = int, default=768)
parser.add_argument('--hidden_dim', type = int, default=300)
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

parser.add_argument('--max_sent_len', type=int, default=200, # according to the pre-trained transformer-xl
                    help='max content length for each text, if set to 0, then the max length has no constrain')
parser.add_argument('--num_heads', nargs='+', type=int, default=[3, 3, 3, 3],
                    help='number of heads:[n_local,n_global,n_speaker,n_listener], default=[3,3,3,3]')
parser.add_argument('--mem_len', type=int, default=1000, help='max memory length')
parser.add_argument('--windowp', type=int, default=5, help='local attention window size')
parser.add_argument('--attn_type', type = str, choices=['uni','bi'], default='bi')

parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

parser.add_argument('--dataset_name', default='EmoryNLP', choices=['IEMOCAP', 'MELD','DailyDialog','EmoryNLP'], type= str,
                                        help='dataset name, IEMOCAP or MELD or DailyDialog or EmoryNLP')

parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate')

parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')

parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')

parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')

parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')


args = parser.parse_args()
print(args)
seed_everything()

n_epochs = args.epochs
batch_size = args.batch_size
if args.basemodel == 'transfo_xl':
    train_loader, valid_loader, test_loader,speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_transfo_xl(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
elif args.basemodel in ['xlnet', 'xlnet_dialog']:
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_xlnet(
        dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)
# renew_cnt = [0 for i in range(len(speaker_vocab['itos']))]
n_classes = len(label_vocab['itos'])

all_utils = []
for mem_len in  [100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300]:
    mem_utils = []
    for dataset in tqdm(test_loader):
        mems = None
        speaker_mask = None
        window_mask = None
        for data in dataset:
            # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
            content_ids, label, content_mask, content_lengths, speaker_ids, _ = data
            label = label.detach().numpy().tolist()

            if mems == None:
                mems = content_mask[:,-mem_len:]
            else:
                mems = torch.cat([mems, content_mask], dim = 1)[:,-mem_len:]
            for i,l in enumerate(label):
                if l != -1:
                    total_use = torch.sum(mems[i])
                    mem_utils.append(float(total_use)/mems.size()[1])

    all_utils.append(np.mean(mem_utils))

print(all_utils)