# -*- coding: utf-8 -*-
"""
script for training GPT-2 on wikitext-103

Created on Thu Mar 18 16:03:16 2021

@author: karmeni1
"""

import argparse
import os
import time
from datetime import timedelta
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_cosine_schedule_with_warmup



# ============================= #
# ===== DATASET MANAGMENT ===== #
# ============================= #


# split dataset into chunks of 1024 tokens
def chunk(l, n, equal_len=True, bos=None, eos=None):
    n = max(1, n)

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


# ============================ #
# ===== TRAINING CLASSES ===== #
# ============================ #


# create config dicts
class TrainConfig(object):
    
    def __init__(self, max_epochs, bs):
    
        self.max_epochs = max_epochs
        self.batch_size = bs


class EvalConfig(object):
    
    def __init__(self, es_patience, bs):
    
        self.es_patience = es_patience
        self.batch_size = bs


class Experiment(object):
    
    """
    attributes:
        
        .x_train: batched training inputs, 
                  shape = (batch_size, batch_id, sequence_length)
    """
    
    def __init__(self,
                 model=None, 
                 x_train=None, x_eval=None, x_test=None,
                 train_config=None, eval_config=None, 
                 optimizer=None,
                 lr_scheduler=None,
                 model_name=None,
                 savedir=None,
                 logdir=None):
        
        self.x_train = x_train
        self.x_eval = x_eval
        self.x_test = x_test
        self.model = model
        self.optim = optimizer
        self.cfg_train = train_config
        self.cfg_eval = eval_config
        self.n_epochs
        self.loss
        self.log = {
            "train_loss": [],
            "eval_loss": [],
            }
        
    def start(self, do_test=False):
        
        # initialize early stopping class 
        early_stopping = self.EarlyStopping(model=self.model,
                                            patience=self.cfg_eval.patience)
        
        print("Starting training...")
        
        # loop over epochs
        for n in list(range(0, self.max_epochs)):
            
            print("Epoch {}/{} ({}%).".format(n, self.max_epochs, 
                                             (n/self.max_epochs)*100))
            
            start_time = time.perf_counter()

            # run .train and .evaluate routines
            train_loss = self.train(x=self.x_train, labels=self.x_train)
            val_loss = self.evaluate(x=self.x_eval, labels=self.x_eval)
            
            # check if best loss is reached and save model weights if so
            early_stopping.check(epoch_id=n, 
                                 current_eval_loss=val_loss)
            
            # measure time after train and eval routines have finished
            end_time = time.perf_counter()
            
            # format elapsed time as HH:MM:SS (and round the seconds)
            elapsed = str(timedelta(seconds=round(end_time-start_time)))
            print("Epoch {} took {} (HH:MM:SS)".format(n, elapsed))
            
            if early_stopping.improvement:
                
                self.save_checkpoint()
            
            if early_stopping.stop:
                
               print("No improvement after {} consec. epochs"
                     .format(early_stopping.patience))
               print("Exiting training at epoch {}..."
                     .format(n))
                
               break
           
        if do_test:
            
            print("Evaluating model on test set ...")
            
            test_loss = self.evaluate(x=self.test_ds, labels=self.test_ds)

        
    def train(self, x, labels):
        """
        .train(x, labels, x_eval, labels_eval) calls .train_step() on all 
        batches in the training dataset (x).
        """            
        
        # set to train mode
        self.model.train()
        
        # assume batch id is in second dimension
        n_batches = x.shape[1]  # shape = (batch_size, batch_id, sequence_len)
        
        # set loss to 0 at the start of the epoch
        loss = 0
        
        # loop over training samples
        for batch_idx in list(range(0, n_batches)):
            
            batch_loss = self.train_step(x_batch=x[:, batch_idx, :])
            loss += batch_loss.cpu().numpy()
            
            self.log["train_loss"].append(loss)
            
            
    def train_step(self, x_batch):
        """
        .train_step() performs a single single trianing step, 
        from input to output and a backward pass. It assumes HuggingFace syntax
        where the call to output = self.model(x=x, labels=x) shifts the labels
        <labels> and returns language modeling loss for the sequence 
        (accessible in output.loss).
        """
        # compute predictions for all tokens in a chunk
        output = self.model(input_ids=x_batch, labels=x_batch)
        
        # do the backprop
        output.loss.backward()
        
        # weight update
        self.optimizer.step()
        
        # clear the gradients
        self.model.zero_grad()
        
        return output.loss.cpu()
        
    def evaluate(self, x, labels):
        
        self.model.eval()
        
        with torch.no_grad():
            
            # assume batch id is in second dimension
            n_batches = x.shape[1]
            
            # set loss to 0 at the start of the epoch
            loss = 0
            
            # loop over training samples
            for batch_idx in list(range(0, n_batches)):
                
                x_batch = x[:, batch_idx, :]
                
                batch_loss = self.model(input_ids=x_batch, labels=x_batch)
                
                loss += batch_loss.cpu().numpy()
                
                self.log["eval_loss"].append(loss)

    def save_checkpoint(self):
         
        full_fname = os.path.join(self.savedir, self.model_name)
        
        print("Saving checkpoint as:\n{}".format(full_fname))
        
        torch.save(self.model.cpu().state_dict(), 
                   full_fname)

    def EarlyStopping(object):
        
        """
        EarlyStopping() is a helpher class that keeps track of evaluation loss.
        EarlyStopping.check() halts Experiment().start() subroutine if evaluation 
        set loss has not decreased for n == EarlyStopping().patience epochs in a row.
        """
        
        def __init__(self, model, patience):
            
            self.model = model     # placeholder for self.model object form the Experiment() class
            self.best_score = None
            self.patience = patience
            self.improvement = False
            self.counter = 0
            self.previous = None
            self.when = None
            self.stop = False
        
        def check_and_save(self, epoch_id, current_val_loss):
            
            """
            EarlyStopping().check(epoch_id[int], current_eval_loss[float]) keeps
            track of the eval loss and halts training at epoch <epoch_id> if 
            <current_eval_loss> has not decreased in <EarlyStopping().patience>
            consecutive epochs.
            """
            
            while self.counter <= self.patience:
                
                # check if lowest loss was reached
                if (self.best_score is None) or (self.best_score > current_val_loss):
                    
                    self.best=current_val_loss # log current loss as best
                    self.counter = 0           # start counting again
                    self.improvement = True
                    self.save_checkpoint()
                
                # check if loss decreased relative from previous epoch, but
                # has not yet beaten the best loss
                elif (self.previous > current_val_loss) & (self.best < current_val_loss):
                
                    self.counter = 0 # reset counter and keep monitoring
                    self.improvement = True
                    
                elif (self.best < current_val_loss) & (self.previous < current_val_loss):
                    
                    self.counter += 1 # current_loss has increased, start counting
                    self.improvement = False
                
                # print some feedback
                print("Epoch {}".format(epoch_id))
                print("Current loss: {}".format(current_val_loss))
                print("Best loss: {}".format(self.best))
                print("Continuing training ...")
                
                # store current loss for next round check
                self.previous = current_val_loss
            
            self.when=epoch_id
            
            return self.counter


# ======================== #
# ===== RUNTIME CODE ===== #
# ======================== #

def runtime_code():
    
    # collect input arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_ds", type=str,
                        help="path to training dataset")
    parser.add_argument("--val_ds", type=str,
                        help="path to validation set")
    parser.add_argument("--test_ds", type=str,
                        help="path to tokenized file")
    parser.add_argument("--model_name", type=str,
                        help="filename for saving model checkpoint to --savedir")
    parser.add_argument("--seed", type=int,
                        help="seed used in torch.manual_seed() for reproducibility")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        help="device used for training")
    parser.add_argument("--batch_size", type=int,
                        help="batch size to use in training")
    parser.add_argument("--max_epochs", type=int,
                        help="maximum number of trainin epochs (iterations)")
    parser.add_argument("--lr", type=int,
                        help="starting learning rate")
    parser.add_argument("--num_lr_warmup_steps", type=int,
                        help="number of consecutive epochs for which learning" 
                        "rate is increased linearly")
    parser.add_argument("--es_patience", type=int,
                        help="nr of consecutive epochs to wait for decreasing loss" 
                        "before stopping training")
    parser.add_argument("--savedir", type=str,
                        help="path where the model weights will be stored")
    parser.add_argument("--logdir", type=str,
                        help="path where the model weights will be stored")
    
    args = parser.parse_args()
    
    # use cuda if available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    # ===== PREPARE DATASET ===== #
    
    # uncomment this for development
    inp = os.path.join("C:\\", "users", "karmeni1", "project", "lm-mem", "data",
                            "wikitext-103", "wiki.test.tokens")
    
    with open(inp, "r", encoding="utf-8") as file_handle:
        
        lines = file_handle.readlines()
        file_handle.close()
    
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # define dataset size
    
    # text is already tokenized so spliting at empty spaces should work
    toks = tokenizer.tokenize("".join(lines))
    
    
    inputs=list(chunk(toks, 1024, equal_len=True))
    
    # create input indices
    input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in inputs]
    
    # now create the labels 
    # shape = (batch_size, batch_id, sequence_length)
    train_ds = torch.tensor(data=batchify(np.asarray(input_ids), bs=5),
                            dtype=torch.long)
    train_ds.to(device)

    eval_ds = None

    test_ds = None
    
    # ==== INITIALIZE CONFIG AND EXPERIMENT CLASSES ===== #
    
    # initialize model with default config and move to device
    torch.manual_seed(args.seed)
    model = GPT2LMHeadModel(GPT2Config()).to(device)
    
    # declare optimizer
    optimizer=AdamW(params=model.parameters(), 
                    lr=args.lr, 
                    betas=(0.9, 0.999))
    
    # initilize learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=args.max_epochs,
                                                num_cycles=0.5,
                                                last_epoch=-1)
    
    exp = Experiment(model=model.to(device),
                     train_ds=None,
                     eval_ds=None,
                     test_ds=None,
                     train_config=TrainConfig(bs=args.batch_size,
                                              n_epochs=args.n_epochs),
                     eval_config=EvalConfig(bs=args.batch_size,
                                            patience=args.es_patience),
                     optimizer=optimizer,
                     lr_scheduler=scheduler,
                     model_name=args.model_name,
                     savedir=args.savedir,
                     logdir=args.logdir)
    
    # ===== TRAINING ROUTINE =====#
    exp.start(do_test=True)

if __name__ == "__main__":

    runtime_code()