# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:03:44 2021

"""

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import argparse
import os

class WikiTextDataset(Dataset):
    
    def __init__(self, tokenizer):
        
        self.x = None  # input samples
        self.tokenizer = tokenizer
    
    # split dataset into chunks of 1024 tokens
    def chunk_list(self, l, n, equal_len=True, bos=None, eos=None):
        
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
    
    
    def load_and_retokenize_txt(self, path, retokenize=False, 
                                save_retokenized=None,
                                sequence_length=None):
            

        with open(path, "r", encoding="utf-8") as file:
            
            lines = file.readlines()
        
        if retokenize:
            
            # retokenize the entire test set
            print("Retokenizing {}".format(path))
            tokens = []
            ids = []
            for l in tqdm(lines, desc="line"):
                tokens += self.tokenizer.tokenize(l)
                ids += self.tokenizer.encode(l)
                
            if save_retokenized:
                
                # save tokens as .json
                print("Saving {}".format(save_retokenized))
                with open(save_retokenized, "w") as fname:
                    fname.writelines(tokens)
                    
                # save tokens as .json
                print("Saving {}".format(save_retokenized + "ids"))
                with open(save_retokenized + "_ids", "w") as fname:
                    fname.writelines(ids)
                    
        else:
            
            tokens = lines
        
        if sequence_length is not None:
            # split list of input tokens into list of elements of size max_len
            print("Chunking samples in self.x to length of {}".format(sequence_length))
            self.x = list(self.chunk_list(tokens, 
                                          n=sequence_length,
                                          bos=self.tokenizer.bos_token_id,
                                          eos=self.tokenizer.eos_token_id))
        elif sequence_length is None:
            print("sequence_length == None, not populating self.x")
        

    def __len__(self):
        
        return len(self.x)
    
    def __getitem__(self, index):
        
        return torch.tensor(self.x[index])


# wrapper function called when script is called
def runtime_code():
     
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_tokens", type=str,
                        help="path to training dataset")
    parser.add_argument("--valid_tokens", type=str,
                        help="path to validation dataset")
    parser.add_argument("--test_tokens", type=str,
                        help="path to test dataset")
    parser.add_argument("--train_set_sizes", type=str,
                        help="size of the training set (in M tokens)")
    parser.add_argument("--train_tokenizer", action="store_true",
                         help="boolean, whether or not to train BPE tokenizer" 
                         "on wikitext103 (default: false)")
    parser.add_argument("--tokenizer_train_tokens", type=str)
    parser.add_argument("--tokenizer_savedir", type=str,
                        help="folder where merges.txt and vocab.txt are saved")
    parser.add_argument("--savedir", type=str, 
                        help="path to where the tokenized dataset is saved")
    
    args = parser.parse_args()
    
    # load the pretrained tokenizer or traine one from scracth
    # max len is not crucial here as we only want to retokenize the dataset, not
    # create the final sequences
    if os.path.exists(args.tokenizer_savedir) and os.path.isdir(args.tokenizer_savedir):
        
        if not os.listdir(args.tokenizer_savedir) and args.train_tokenizer:
            
                    print("Training BPE tokenizer...")
                    tokenizer = ByteLevelBPETokenizer()
                    tokenizer.train(files=[args.tokenizer_train_tokens, args.valid_tokens, args.test_tokens],
                                    vocab_size=28439, 
                                    min_frequency=2, 
                                    special_tokens=["<|endoftext|>", "<pad>"])

                    # Save files to disk
                    print("Saving merges.txt and vocab.json to {}".format(args.tokenizer_savedir))
                    tokenizer.save_model(args.tokenizer_savedir)
                    
        elif not os.listdir(args.tokenizer_savedir) and not args.train_tokenizer:
            raise ValueError("Tokenizer directory is empty. Use --train_tokenizer.)
                             
        else:    
            tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_savedir, max_len=1024)
    else:
        print("{} directory doesn't exist".format(tokenizer.savedir))
    
    
    # now retokenize wikitext and save
    train_ds = WikiTextDataset(tokenizer=tokenizer)
    
    # training set
    train_ds.load_and_retokenize_txt(path=args.train_tokens,
                                     retokenize=True,
                                     save_retokenized=args.train_tokens.replace("tokens", "tokens.bpe"))
    
    # validation set
    eval_ds = WikiTextDataset(tokenizer=tokenizer)
    eval_ds.load_and_retokenize_txt(path=args.val_tokens,
                                    retokenize=True,
                                    save_retokenized=args.valid_tokens.replace("tokens", "tokens.bpe"))
    
    # validation set
    eval_ds = WikiTextDataset(tokenizer=tokenizer)
    eval_ds.load_and_retokenize_txt(path=args.test_tokens,
                                    retokenize=True,
                                    save_retokenized=args.valid_tokens.replace("tokens", "tokens.bpe"))
    
if __name__ == "__main__":
    
    runtime_code()
