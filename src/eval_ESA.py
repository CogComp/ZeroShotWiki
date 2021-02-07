import time
from load_customized_data_ESA import load_dataset_and_labelnames
# from ESA import load_ESA_sparse_matrix, divide_sparseMatrix_by_list_row_wise, multiply_sparseMatrix_by_list_row_wise
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from scipy.sparse import vstack
import numpy as np
# from operator import itemgetter
from scipy.special import softmax
from scipy import sparse
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from math import ceil
import numpy as np
from multiprocessing import current_process
import os


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain the .tsv files for the task.")
parser.add_argument("--label_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The label name file. Multiple words for the same label should be split by space")
parser.add_argument("--esa_path",
                    default='../model/ESA_WikiCate/',
                    type=str,
                    required=False,
                    help="Directory that stores ESA data")
parser.add_argument("--multi_label",
                    action='store_true',
                    help="Use multi-label f1")
parser.add_argument("--threshold",
                    default=0.1,
                    type=float,
                    help="The threshold used for multi-label prediction")
parser.add_argument("--n_jobs",
                    default=1,
                    type=int,
                    required=True,
                    help="Number of parallel jobs")

args = parser.parse_args()

vocab_path = os.path.join(args.esa_path, 'vocab.txt')
idf_path = os.path.join(args.esa_path, 'idf_vec.npy')
all_texts, all_labels, all_word2DF, labelnames, wiki_idf = load_dataset_and_labelnames(args.data_path, args.label_path, vocab_path, idf_path)

ESA_sparse_matrix = sparse.load_npz(os.path.join(args.esa_path, 'ESA_tfidf_csr_truncated.npz'))


def mul_sp_row(mat, lis):
    arr = sparse.csr_matrix(np.expand_dims(np.array(lis),axis=1))
    # print(arr.shape, mat.shape)
    return mat.multiply(arr)

def div_sp_row(mat, lis):
    arr = sparse.csr_matrix(np.expand_dims((1/np.array(lis)),axis=1))
    # print(arr.shape, mat.shape)
    return mat.multiply(arr)

def sparse_row_sum(spm):
    vec = sparse.csr_matrix(np.ones([1, spm.shape[0]]))
    # return spm.__rmul__(vec)
    return vec.dot(spm)


def text_idlist_2_ESAVector(idlist, text_bool):
    if text_bool:
        sub_matrix = ESA_sparse_matrix[idlist,:]
        # myvalues = [all_word2DF.get(id) for id in idlist]
        # weighted_matrix = div_sp_row(sub_matrix, myvalues)
        myvalues = wiki_idf[idlist]
        weighted_matrix = mul_sp_row(sub_matrix, myvalues)
        
        return sparse_row_sum(weighted_matrix)
    else: #label names
        sub_matrix = ESA_sparse_matrix[idlist,:]
        # return  sparse.csr_matrix(sub_matrix.sum(axis=0))
        return sparse_row_sum(sub_matrix)


    
def ESA_cosine():
    label_veclist = []
    for i in range(len(labelnames)):
        labelname_idlist = labelnames[i]
        '''label rep is sum up all word ESA vectors'''
        label_veclist.append(text_idlist_2_ESAVector(labelname_idlist, False))
    labels = all_labels[0]
    docs = all_texts[0]
    sample_size = len(labels)
    print('total test size:', sample_size)
    co=0
    
    label_stack = sparse.vstack(label_veclist)
    
    
    def ESA_cosine_dist(chunk): 
        current = current_process()
        pred_labels = []
        cosine_values = []
        
        if current._identity[0]==1:
            for text_idlist in tqdm(chunk, desc="Iteration"):
                text_vec = text_idlist_2_ESAVector(text_idlist, True)
                cos_array=cosine_similarity(text_vec, label_stack)
                max_id = np.argmax(cos_array, axis=1)
                # gold_label = labels[sample_idx]
                pred_labels.append(max_id[0])
                cosine_values.append(cos_array)
        else:
            for text_idlist in chunk:
                text_vec = text_idlist_2_ESAVector(text_idlist, True)
                cos_array=cosine_similarity(text_vec, label_stack)
                max_id = np.argmax(cos_array, axis=1)
                # gold_label = labels[sample_idx]
                pred_labels.append(max_id[0])
                cosine_values.append(cos_array)
        
        return pred_labels, cosine_values 
    
    start = time.time()
    chunk_size = ceil(len(docs) / args.n_jobs)
    chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
    
    result = Parallel(n_jobs=args.n_jobs)(delayed(ESA_cosine_dist)(chunk) for chunk in chunks)
    pred = np.concatenate([x[0] for x in result])
    cosine_values = np.concatenate([x[1] for x in result])
    cosine_values = np.reshape(cosine_values, [sample_size, len(labelnames)])
    end = time.time()
    print("time: {}".format(end-start))
            
    if args.multi_label:
        if args.threshold is None:
            raise ValueError("Please specify the threshold for multi-label prediction")
        pred = np.array(cosine_values>args.threshold, dtype=int)
        pred_last_col = np.zeros([pred.shape[0], 1])
        for i in range(pred.shape[0]):
            if not pred[i].any():
                pred_last_col[i] = 1
        pred = np.hstack([pred, pred_last_col])
        
        true = np.zeros(pred.shape)
        for i in range(len(labels)):
            true[i, labels[i]] = 1
        acc = f1_score(true, pred, average='weighted')
        print('Multi-label f1:', acc)
        # print(multilabel_confusion_matrix(true, pred))
    
    else:      
        hit_size = 0
        for i in range(sample_size):
            if pred[i] == labels[i][0]:
                hit_size+=1
        
        acc = hit_size/sample_size
        print('acc:', acc)


if __name__ == '__main__':
    ESA_cosine()
