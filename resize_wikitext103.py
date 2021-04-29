# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:02:32 2021

@author: karmeni1
"""

import argparse
import os
from ast import literal_eval

# wrapper for loading datasets
def load_dataset(path):
    
    # make sure it handles "~" in path string correctly
    path = os.path.expanduser(path)
    
    with open(path, "r", encoding="utf-8") as file_handle:
        
        lines = file_handle.readlines()
        file_handle.close()

    return lines


# ===== PREPARE DATASET ===== #

# wrapper function called when script is called
def runtime_code():
     
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datadir", type=str,
                        help="path to files with wikitext103 tokens")
    parser.add_argument("--train_ds", type=str,
                        help="path to training dataset")
    parser.add_argument("--train_set_sizes", type=str,
                        help="size of the training set (in M tokens)")
    parser.add_argument("--savedir", type=str, 
                        help="path to where the tokenized dataset is saved")
    
    args = parser.parse_args()
    
    #start_time = time.perf_counter()
    
    # datasets are already split into words/pre-tokenized (but not for GPT-2)
    print("Loading datasets")
    wiki_train = load_dataset(path=os.path.join(args.datadir, args.train_ds))
    
    # resize train set to selected nr of tokens (in mio)
    n_tokens = int(args.train_set_size*1e6)
    print("Resizing train set to {} tokens".format(n_tokens))
    
    # join wiki subsections and then split where each token is a list element
    # dataset is pretokenized
    wiki_train = "".join(wiki_train).split(" ")
    
    
    start = 0
    for size in literal_eval(args.train_set_sizes):
        
        end = start + int(size* 1e6)
        
        print("Resizing to {} tokens".format(end-start))
        
        wiki_train_resized = "".join(wiki_train[start:end])
        
        outfile = os.path.join(os.savedir, "wiki.train.tokens_{}".format(size), "w")
        
        with open(outfile, "w") as f:
            
            f.writelines(wiki_train_resized)
    
        start = end    
    
# CALL THE CODE ABOVE
if __name__ == "__main__":

    runtime_code()
