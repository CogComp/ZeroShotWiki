import json
import codecs
# import nltk
from collections import defaultdict
import numpy as np


def text2idlist(token_list, word2id):
    idlist=[]
    for word in token_list:
        id=word2id.get(word)
        if id is not None: # if word was not in the vocabulary
            idlist.append(id)
    return idlist

def load_customized_dataset(data_path, word2id):
     
    
    all_texts=[]
    all_labels=[]
    all_word2DF=defaultdict(int)
    max_sen_len=0

    print('loading file:', data_path, '...')

    texts=[]
    # text_masks=[]
    labels=[]
    missing = []
    readfile=codecs.open(data_path, 'r')
    line_co=0
    for i, line in enumerate(readfile):
        parts = line.strip().split('\t', 1)
        if len(parts)==2:
            # label_id = int(parts[0])
            label_id = parts[0].strip().split()
            '''truncate can speed up'''
            text_wordlist = parts[1].strip().lower().split()[:100]#[:30]
            text_len=len(text_wordlist)
            if text_len > max_sen_len:
                max_sen_len=text_len
            text_idlist=text2idlist(text_wordlist, word2id)
            if len(text_idlist) >0:
                texts.append(text_idlist)
                # labels.append(label_id)
                labels.append([int(lid) for lid in label_id])
                idset = set(text_idlist)
                for iddd in idset:
                    all_word2DF[iddd]+=1
            else:
                missing.append(i)
                texts.append([0])
                # labels.append(label_id)
                labels.append([int(lid) for lid in label_id])
                continue
        

        line_co+=1

    all_texts.append(texts)
    all_labels.append(labels)
    print('load text successfully')
    return all_texts, all_labels, all_word2DF


def load_labels_customized(label_path, word2id):
    
    texts=[]
    readfile=codecs.open(label_path, 'r')
    for line in readfile:        
        wordlist = line.strip().replace('&', ' ').lower().split()

        text_idlist=text2idlist(wordlist, word2id)
        if len(text_idlist) >0:
            texts.append(text_idlist)

    print('load label names successfully, totally :', len(texts), 'label names')
    
    return texts


def load_dataset_and_labelnames(data_path, label_path, vocab_path, idf_path):
    word2id = {}
    with open(vocab_path, 'r') as f:
        for i, line in enumerate(f):
            word2id[line.strip()] = i
    wiki_idf = np.load(idf_path)
    
    all_texts, all_labels, all_word2DF = load_customized_dataset(data_path, word2id)
    labelnames = load_labels_customized(label_path, word2id)
    return all_texts, all_labels, all_word2DF, labelnames, wiki_idf