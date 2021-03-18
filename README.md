# DialogXL
This repo contains the PyTorch implementaion for AAAI-2021 accepted paper DialogXL: All-in-One XLNet for Multi-Party Conversation Emotion Recognition.

## Update 2021/3/18
Thanks to my co-author Junqing (github id: [Digimonseeker](https://github.com/Digimonseeker)), we have now updated our code to make it compatible with the latest version of huggingface's Transformers. Now you can run our code with **Transformers 4.3.3**, and the pre-trained XLNet can be downloaded from https://huggingface.co/xlnet-base-cased.

The hyper-parameters we previously provided still work well, but the well-trained DialogXL models we uploaded may be unable to run in this code version.

## Requirements
* Python 3.6
* PyTorch 1.4.0
* [Transformers](https://github.com/huggingface/transformers) 4.3.3
* scikit-learn 0.23.1
* CUDA 10.0

## Preparation

### Datasets
We have provided the preprocessed datasets of IEMOCAP, MELD, DailyDialog and EmoryNLP in `/data/`. For each dataset, there are 3 json files containing dialogs for train/dev/test set. Each dialog contains utterances
 and their corresponding speaker information and emotion labels.
 
### Pre-trained XLNet
You should download the pytorch version pre-trained 
`xlnet-base` model and vocabulary from the link provided by huggingface. 
Then change the value of parameter `--bert_model_dir` and `--bert_tokenizer_dir` to the directory of the bert model.

## Evaluate
You can download our well-trained DialogXL for each dataset from 
https://drive.google.com/drive/folders/1ONGBbo9r9S49i5bz4jgrC79txQHNVZv2?usp=sharing
or https://pan.baidu.com/s/1SYbpCI4LTzUnYbRKzPvijg 提取码 jpfe

Then put them to `/DialogXL/saved_models/`

You can run the following codes to get results very close to those reported in our paper (we report the average of 5 random runs in paper):

For IEMOCAP (test F1: 65.88): 
`python eval.py --dataset_name IEMOCAP --max_sent_len 200 --mem_len 900 --windowp 10 --num_heads 2 2 4 4 --modelname IEMOCAP_xlnet_dialog`

For MELD (test F1: 62.67):
`python eval.py --dataset_name MELD --max_sent_len 300 --mem_len 400 --windowp 5 --num_heads 5 5 1 1 --modelname MELD_xlnet_dialog`

For DailyDialog (test F1: 55.67): 
`python eval.py --dataset_name DailyDialog --max_sent_len 300 --mem_len 450 --windowp 10 --num_heads 1 2 5 4 --dropout 0.3 --modelname DailyDialog_xlnet_dialog
`

For EmoryNLP (test F1: 36.27): 
`python eval.py --dataset_name EmoryNLP --max_sent_len 150 --mem_len 400 --windowp 5 --num_heads 1 2 4 5 --modelname EmoryNLP_xlnet_dialog`


## Training
You can also train the models with the following codes:

For IEMOCAP: 
`python run.py --dataset_name IEMOCAP --max_sent_len 200 --mem_len 900 --windowp 10 --num_heads 2 2 4 4 --dropout 0 --lr 1e-5 --epochs 50`

For MELD: 
`python run.py --dataset_name MELD --max_sent_len 300 --mem_len 400 --windowp 5 --num_heads 5 5 1 1 --dropout 0 --lr 1e-6 --epochs 15`

For DailyDialog: 
`python run.py --dataset_name DailyDialog --max_sent_len 300 --mem_len 450 --windowp 10 --num_heads 1 2 5 4 --dropout 0.3 --lr 1e-6 --epochs 20`

For EmoryNLP: 
`python run.py --dataset_name EmoryNLP --max_sent_len 150 --mem_len 400 --windowp 5 --num_heads 1 2 4 5 --dropout 0 --lr 7e-6 --epochs 10`

