from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
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


def get_hypothesis(label_file, simple_hypo=False):
    labelnames = []
    with open(label_file, 'r') as f:
        for i, line in enumerate(f):
            labelnames.append(line.strip().split())
    type2hypothesis = {}
    for i, words in enumerate(labelnames):
        if simple_hypo:
            type2hypothesis[i] = [' or '.join(words)]
        else:
            type2hypothesis[i] = ['it is related with ' + ' or '.join(words)]
        
    return type2hypothesis
        
    

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

    def get_examples_Yahoo_test(self, filename, type2hypothesis, labeling='single'):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        exam_co = 0
        examples=[]

        gold_label_list = []
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==2: # label_id, text
                if labeling=='single':
                    type_index =  int(line[0])
                    gold_label_list.append(type_index)
                elif labeling=='multi':
                    type_index =  [int(x) for x in line[0].split()]
                    gold_label_list.append(type_index)
                else:
                    raise ValueError('Invalid labeling type')
                for i in range(len(type2hypothesis)):
                    hypo_list = type2hypothesis.get(i)
                    for hypo in hypo_list:
                        guid = "test-"+str(exam_co)
                        text_a = line[1]
                        text_b = hypo
                        label = 'not_entailment' # fake label for test
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        exam_co+=1
                line_co+=1

        readfile.close()
        print('loaded size:', line_co)
        return examples, gold_label_list
    
    
    def get_labels(self):
        return ["entailment", "not_entailment"]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    premise_2_tokenzed={}
    hypothesis_2_tokenzed={}
    list_2_tokenizedID = {}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = premise_2_tokenzed.get(example.text_a)
        if tokens_a is None:
            tokens_a = tokenizer.tokenize(example.text_a)
            premise_2_tokenzed[example.text_a] = tokens_a

        tokens_b = premise_2_tokenzed.get(example.text_b)
        if tokens_b is None:
            tokens_b = tokenizer.tokenize(example.text_b)
            hypothesis_2_tokenzed[example.text_b] = tokens_b

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens_A = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_A = [0] * len(tokens_A)
        tokens_B = tokens_b + ["[SEP]"]
        segment_ids_B = [1] * (len(tokens_b) + 1)
        tokens = tokens_A+tokens_B
        segment_ids = segment_ids_A+segment_ids_B


        input_ids_A = list_2_tokenizedID.get(' '.join(tokens_A))
        if input_ids_A is None:
            input_ids_A = tokenizer.convert_tokens_to_ids(tokens_A)
            list_2_tokenizedID[' '.join(tokens_A)] = input_ids_A
        input_ids_B = list_2_tokenizedID.get(' '.join(tokens_B))
        if input_ids_B is None:
            input_ids_B = tokenizer.convert_tokens_to_ids(tokens_B)
            list_2_tokenizedID[' '.join(tokens_B)] = input_ids_B
        input_ids = input_ids_A + input_ids_B


        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
        
        
class evaluation_runner():
    
    def __init__(self, eval_data, eval_dataloader, eval_label_list, eval_batch_size, num_classes, labeling):
        
        self.eval_data = eval_data
        self.eval_dataloader = eval_dataloader
        self.eval_label_list = eval_label_list
        self.eval_batch_size = eval_batch_size
        self.num_classes = num_classes
        self.labeling = labeling
        if self.labeling not in ['single', 'multi']:
            raise ValueError('Invalid labeling type')
    
    def get_prediction_from_entailment(self, entail_prob):
        
        assert (entail_prob.shape[0]/self.num_classes).is_integer()
        assert self.labeling=='single'
        num_examples = int(entail_prob.shape[0]/self.num_classes)
        preds = []
        for i in range(num_examples):
            probs = entail_prob[i*self.num_classes:(i+1)*self.num_classes]
            preds.append(np.argmax(probs))
        return np.array(preds)

    def run_eval(self, model, device, use_segment_id, verbose=False):

        model.eval()
        
        if verbose:
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(self.eval_data))
            logger.info("  Batch size = %d", self.eval_batch_size)
            iterator = tqdm(self.eval_dataloader, desc="Iteration")
        else:
            iterator = self.eval_dataloader

        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        # print('Evaluating...')
        
        for input_ids, input_mask, segment_ids, label_ids in iterator:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                if use_segment_id:
                    logits = model(input_ids, input_mask, segment_ids, labels=None)
                else:
                    logits = model(input_ids, input_mask, labels=None)
            logits = logits[0]

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        pred_probs = softmax(preds,axis=1)[:,0]
        
                
        if self.labeling=='single':
            pred_label = self.get_prediction_from_entailment(pred_probs)
            micro_f1 = f1_score(self.eval_label_list, pred_label, average='micro')     
            return micro_f1
        else:
            pred_probs = pred_probs.reshape([len(self.eval_label_list), self.num_classes])
            pred = (pred_probs>0.5).astype(int)
            pred_last_col = np.zeros([pred.shape[0], 1])
            for i in range(pred.shape[0]):
                if not pred[i].any():
                    pred_last_col[i] = 1
            pred = np.hstack([pred, pred_last_col])

            true = np.zeros(pred.shape)
            for i in range(len(self.eval_label_list)):
                true[i, self.eval_label_list[i]] = 1
            weighted_f1 = f1_score(true, pred, average='weighted')
            return weighted_f1


        
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def main():
    
    blockPrint()
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data file. Should contain the .tsv files for the task.")
    parser.add_argument("--label_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The label name file. Multiple words for the same label should be split by space")
    
    ## Other parameters
    parser.add_argument("--model_path",
                        default='../model/TE_WikiCate',
                        type=str,
                        required=False,
                        help="The model name file.")    
    parser.add_argument("--simple_hypo",
                        action='store_true',
                        help="Whether to use word-like hypotheses")
    parser.add_argument("--multi_label",
                        action='store_true',
                        help='Whether the data is multi-labeled')
    
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
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
    args = parser.parse_args()


    type2hypothesis = get_hypothesis(args.label_path, args.simple_hypo)
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    processor = DataProcessor()

    label_list = processor.get_labels() #[0,1]
    num_labels = len(label_list)


    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_TRANSFORMERS_CACHE), 'distributed_{}'.format(args.local_rank))
    pretrain_model_dir = args.model_path
    model = BertForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_unseen_acc = 0.0
    max_dev_unseen_acc = 0.0
    max_dev_seen_acc = 0.0
    max_overall_acc = 0.0

    '''load test set'''

    if args.multi_label:
        labeling = 'multi'
    else:
        labeling = 'single'
    test_examples, test_label_list = processor.get_examples_Yahoo_test(args.data_path, type2hypothesis, labeling)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)

    test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    
    test_runner = evaluation_runner(test_data, test_dataloader, test_label_list, args.eval_batch_size, len(type2hypothesis), labeling)

    model.eval()
    enablePrint()
    print('Testing...')    
    f1 = test_runner.run_eval(model, device, use_segment_id=True, verbose=True)
    print("F1 score: {}".format(f1))
    

        

if __name__ == "__main__":
    main()

