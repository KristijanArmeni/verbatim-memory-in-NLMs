# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:02:32 2021

@author: karmeni1
"""

import argparse
import os
import time
from datetime import timedelta
import numpy as np
import json
from transformers import AutoTokenizer


# ============================= #
# ===== DATASET FUNCTIONS ===== #
# ============================= #

# wrapper for loading datasets
def load_dataset(path):
    
    # make sure it handles "~" in path string correctly
    path = os.path.expanduser(path)
    
    with open(path, "r", encoding="utf-8") as file_handle:
        
        lines = file_handle.readlines()
        file_handle.close()

    return lines


# split dataset into chunks of 1024 tokens
def chunk(l, n, equal_len=True, bos=None, eos=None):
    
    n = max(1, n)
    
    if (bos is not None) and (eos is not None):
        
        # if bos and eos need to be added adjust selected n to not exceed
        # the allowed maximum
        print("Adding bos and eos setting {} to {}".format(n, n-2))
        n -= 2
    
    if equal_len:
        total_len = (len(l)//n)*n
    else:
        total_len = len(l)
    
    output_list =([bos] + l[i:i+n] + [eos] for i in range(0, total_len, n))
    
    return output_list


# wrapper for batching the data
def batchify(x, bs):
    """
    batchifies input array into a new np.array of dimension
    (batch_size, -1, *)
    """
    nb = len(x)//bs
    
    # trim the remainder samples along dimension 1
    print("Batching data, using batch size of {}".format(bs))
    print("Trimming {} remainer sample points".format(x.shape[0]-(bs*nb)))
    xtmp = x[0:(nb*bs)]
    
    # define the shape (batch_size, training samples, *)
    newshape = tuple([bs, -1] + [d for d in x.shape[1::]])
    
    return np.reshape(a=xtmp, newshape=newshape)


# wrapper for resizing
def resize(toks, n_tokens):
    
    if n_tokens < len(toks):   
        out = toks[0:n_tokens]
    else:
        raise ValueError("Cannot resize, new size larger than existing size")
    
    return out


# ===== PREPARE DATASET ===== #

# wrapper function called when script is called
def runtime_code():
     
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datadir", type=str,
                        help="path to files with wikitext103 tokens")
    parser.add_argument("--train_ds", type=str,
                        help="path to training dataset")
    parser.add_argument("--val_ds", type=str,
                        help="path to validation set")
    parser.add_argument("--test_ds", type=str,
                        help="path to tokenized file")
    parser.add_argument("--train_set_size", type=float,
                        help="size of the training set (in M tokens)")
    parser.add_argument("--sequence_len", type=int,
                        help="sequence length (in tokens) used in training and evaluation")
    parser.add_argument("--savedir", type=str, 
                        help="path to where the tokenized dataset is saved")
    
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    
    # datasets are already split into words/pre-tokenized (but not for GPT-2)
    print("Loading datasets")
    wiki_train = load_dataset(path=os.path.join(args.datadir, args.train_ds))
    wiki_val = load_dataset(path=os.path.join(args.datadir, args.val_ds))
    wiki_test = load_dataset(path=os.path.join(args.datadir, args.test_ds))
    
    # resize train set to selected nr of tokens (in mio)
    n_tokens = int(args.train_set_size*1e6)
    print("Resizing train set to {} tokens".format(n_tokens))
    
    # join wiki subsections and then split where each token is a list element
    # dataset is pretokenized
    wiki_train = "".join(wiki_train).split(" ")
    wiki_train_resized = resize(toks=wiki_train, n_tokens=n_tokens)
    wiki_train = None # clear wiki_train
    
    # initialize tokenizer
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # join the read in lines (they're separated by \n so joining with empty 
    # strings is fine)
    print("Tokenizing train set ...")
    train_toks = tokenizer.tokenize("".join(wiki_train_resized))
    wiki_train_resized = None
    
    print("Tokenizing valid set ...")
    val_toks = tokenizer.tokenize("".join(wiki_val))
    print("Tokenizing test set ...")
    test_toks = tokenizer.tokenize("".join(wiki_test))
    
    # clear for memory
    wiki_val, wiki_test = None, None
    
    # split into chunks of equal lengths, pad at the end and at the beginning
    # with eos
    print("Splitting tokens into chunks of {} tokens".format(args.sequence_len))
    eos = "|<endoftext>|"
    train_toks_chunked = list(chunk(train_toks, args.sequence_len, equal_len=True,
                           eos=eos, bos=eos))
    val_toks_chunked = list(chunk(val_toks, args.sequence_len, equal_len=True,
                         eos=eos, bos=eos))
    test_toks_chunked = list(chunk(test_toks, args.sequence_len, equal_len=True,
                         eos=eos, bos=eos))
    
    # clear for memory
    train_toks, val_toks, test_toks = None, None, None
    
    # create input indices
    train_input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in train_toks_chunked]
    val_input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in val_toks_chunked]
    test_input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in test_toks_chunked]
    
    # clear for memory
    train_toks_chunked, val_toks_chunked, test_toks_chunked = None, None, None
    
    end_time = time.perf_counter()
    
    # print some timing feedback
    elapsed = str(timedelta(seconds=round(end_time-start_time)))
    print("Dataset preparation took {} (HH:MM:SS)".format(elapsed))

    # save tensors
    fname = os.path.join(args.savedir, "ids_train_ds.json")
    with open(fname, 'w') as f:
        json.dump(train_input_ids, f)
    
    # save tensors
    fname = os.path.join(args.savedir, "ids_valid_ds.json")
    with open(fname, 'w') as f:
        json.dump(val_input_ids, f)
        
    # save tensors
    fname = os.path.join(args.savedir, "ids_test_ds.json")
    with open(fname, 'w') as f:
        json.dump(test_input_ids, f)


# CALL THE CODE ABOVE
if __name__ == "__main__":

    runtime_code()
