from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import json
import codecs
import numpy as np
import torch
import nltk
from collections import defaultdict, Counter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss
from scipy.special import softmax
from sklearn.metrics import f1_score

from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)
logger = logging.getLogger(__name__)
  

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



class DataProcessor(object):
      
    def random_category_generator(self, all_categories):
        while True:
            perm_cates = np.random.permutation(all_categories)
            for c in perm_cates:
                yield c
    
    def get_random_neg_types(self, type_list, num_neg, cate_gen):
        neg_types = []
        assert num_neg>0 
        assert type(num_neg)==int
        while True:
            c=next(cate_gen)
            if c not in type_list:
                neg_types.append(c)
                if len(neg_types)==num_neg:
                    return neg_types
                
    def load_categories(self, cate_file):
        
        with open(cate_file, 'r') as f:
            page2content_cate = json.load(f)
        page2cate = defaultdict(list)
        for p, c_dic in page2content_cate.items():
            dist = [d for n, d in c_dic['top-level']]
            indices = np.where(dist==np.min(dist))
            gold_cates = np.array(c_dic['top-level'])[indices]
            gold_cates = [n for n, d in gold_cates]
            gold_cates = [' '.join(n.split('_')) for n in gold_cates]
            page2cate[p] = gold_cates
        
        cate2page = defaultdict(list)
        for p, cates in page2cate.items():
            for c in cates:
                cate2page[c].append(p) 
                
        return page2cate, cate2page
        

    def get_examples_Wikipedia_train(self, data_file, cate_file):
        
        page2cate, cate2page = self.load_categories(cate_file)
        
        readfile = codecs.open(data_file, 'r', 'utf-8')

        line_co=0
        exam_co = 0
        examples=[]
        label_list = []
        
        article_cnt = 0
        cnts = {'pos':0, 'neg':0}
        report_flag=True
        
        all_categories = list(cate2page.keys())
        cate_gen = self.random_category_generator(all_categories)

        history_types = set()
        for line in tqdm(readfile, desc='Preparing training data'):
            try:
                line_dic = json.loads(line)
            except ValueError:
                continue

            text = line_dic.get('text')
            tokens = text.split(' ')
            if len(tokens)<=5:
                continue
            
            if len(tokens)>200:
                text_a = ' '.join(tokens[:200])
            else:
                text_a = text
            
            page_id = line_dic.get('id')
            type_list = page2cate[page_id]
            if len(type_list)==0:
                continue
            
            article_cnt += 1
            pos_types = set()
            '''pos pair'''
            for hypo in type_list:
                guid = "train-"+str(exam_co)
                text_b = hypo
                label = 'entailment' #if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                exam_co+=1
                cnts['pos']+=1
            

            # '''neg pair'''
            
            sampled_type_set = self.get_random_neg_types(type_list, 1, cate_gen)
            for hypo in sampled_type_set:
                guid = "train-"+str(exam_co)
                text_b = hypo
                label = 'not_entailment' #if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                exam_co+=1
                cnts['neg']+=1  

        readfile.close()
        print('loaded size:', exam_co)
        print(cnts)
        
        return examples


    def get_labels(self):
        return ["entailment", "not_entailment"]


    
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    
    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 100000 == 0:
            # logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            pass
        
        tokenized_output = tokenizer(example.text_a, example.text_b, padding='max_length', 
                                     truncation=True, max_length=max_seq_length, return_token_type_ids=True)
        input_ids = tokenized_output['input_ids']
        input_mask = tokenized_output['attention_mask']
        segment_ids = tokenized_output['token_type_ids']
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = label_map[example.label]         
            
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: {}".format(tokenizer.convert_ids_to_tokens(input_ids)))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features
    



def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_file",
                        default="data/wikipedia/tokenized_wiki.txt",
                        type=str,
                        help="Path to tokenized wikipedia articles")
    parser.add_argument("--cate_file",
                        default="data/wikipedia/page2content_cate.json",
                        type=str,
                        help="Path to category structure")
    parser.add_argument("--train_steps",
                        default=1500,
                        type=int,
                        help="Number of train steps for early stopping")
    parser.add_argument("--save_dir",
                        default='model/new_model',
                        type=str,
                        required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--temp_file",
                        default='',
                        type=str,
                        help="Directory to save the tensorized trainingd data")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
              
    train_loader_file = args.temp_file

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    processor = DataProcessor()

    label_list = processor.get_labels() #[0,1]
    num_labels = len(label_list)


    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_TRANSFORMERS_CACHE), 'distributed_{}'.format(args.local_rank))


    model_name = 'bert'
    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer
    
    if os.path.exists(args.save_dir):
        raise ValueError('Finetune model directory already exists')
        
    model = model_class.from_pretrained('bert-base-uncased', num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

    model.to(device)
    
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
        
    # prepare train data
    train_examples = None
    num_train_optimization_steps = None

    if os.path.exists(train_loader_file):
        train_data = torch.load(train_loader_file)
    else:        
        train_examples = processor.get_examples_Wikipedia_train(args.data_file, args.cate_file)

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        
        if train_loader_file:
            torch.save(train_data, train_loader_file)    

    num_train_examples = len(train_data)

    num_train_optimization_steps = int(
         num_train_examples / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
        

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_train_examples)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    
    # print('np', np.random.randint(1,100,size=10))
    # print('torch', torch.randint(1, 100, (10,)))
    # print('native', [random.randint(1,100) for i in range(10)])
    loss_hist = []
    stop_train = False
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for batch in tqdm(train_dataloader, desc="Iteration"):

            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = model(input_ids, input_mask, segment_ids, labels=None)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits[0].view(-1, num_labels), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            
            if global_step==args.train_steps:
                stop_train=True
                break
            
            loss_hist.append(str(float(loss)))
            
            global_step += 1
            
                
        if stop_train:          
            break
    
    if args.save_dir:
        model.module.save_pretrained(args.save_dir)
        
        

if __name__ == "__main__":
    main()

