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
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_cosine_schedule_with_warmup


# ============================= #
# ===== DATASET FUNCTIONS ===== #
# ============================= #

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


# ============================ #
# ===== TRAINING CLASSES ===== #
# ============================ #


# create helper config classes
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
    Main high-level class holding all necessary components for training (model, train_ds etc.)
    and associated methods (.train(), evaluate(), .savecheckpoint() etc.)
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
            "test_ppl": None,
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
            _ = self.train(x=self.x_train.to(self.device))
            val_loss = self.evaluate(x=self.x_eval.to(self.device))
            
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
            
            test_loss = self.evaluate(x=self.test_ds)
            self.log["test_ppl": np.exp(test_loss)]

        
    def train(self, x):
        """
        .train(input_dataset) calls .train_step() on all 
        batches in the training dataset (x).
        """            
        
        # set to train mode
        self.model.train()
        
        # assume batch id is in second dimension
        # shape = (batch_size, batch_id, sequence_len)
        n_batches = x.shape[1]  
        
        # set loss to 0 at the start of the epoch
        loss = 0
        
        # loop over training samples
        for batch_idx in tqdm(range(0, n_batches), desc="train set", unit="batch"):
            
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
        # compute predictions for all sequences in a batch
        output = self.model(input_ids=x_batch, labels=x_batch)
        
        # compute gradients
        output.loss.backward()
        
        # update weights
        self.optimizer.step()
        
        # change learning rate according to chosen scheduler
        self.scheduler.step()
        
        # clear the gradients
        self.model.zero_grad()
        
        return output.loss.cpu()
        
    def evaluate(self, x):
        
        self.model.eval()
        
        with torch.no_grad():
            
            # assume batch id is in second dimension
            n_batches = x.shape[1]
            
            # set loss to 0 at the start of the epoch
            loss = 0
            
            # loop over training samples
            for batch_idx in tqdm(range(0, n_batches), desc="validation set:",
                                  unit="batch"):
                
                x_batch = x[:, batch_idx, :]
                
                batch_loss = self.model(input_ids=x_batch, labels=x_batch)
                
                loss += batch_loss.cpu().numpy()
                
                self.log["eval_loss"].append(loss)

    def save_checkpoint(self, savename=None):
        
        if savename is None:
            savename = os.path.join(self.savedir, self.model_name)
        
        print("Saving checkpoint as:\n{}".format(savename))
        
        torch.save(self.model.cpu().state_dict(), 
                   savename)

    def save_configs(self, full_filename):
        
        pass

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
                
            # check if lowest loss was reached and log it
            if (self.best_score is None) or (self.best_score > current_val_loss):
                
                self.best = current_val_loss # log current loss as best
                self.counter = 0             # start counting again
                self.improvement = True
                self.save_checkpoint()
            
            # if not, increase counter and check for early stopping condition
            elif (self.best < current_val_loss):
                
                self.counter += 1
                self.improvement = False
                
                # check if there's we've reached the threshold of no improvement
                if self.counter >= self.patience:
                    
                    # setting this to True halts Experiment().start()
                    # routine
                    self.stop = True
                    self.when = epoch_id
            
            # print some feedback
            print("Epoch: {}".format(epoch_id))
            print("Current loss: {}".format(current_val_loss))
            print("Best loss: {}".format(self.best))
            print("Counter: {}".format(self.counter))


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
    parser.add_argument("--train_batch_size", type=int,
                        help="batch size to use in training ddataset")
    parser.add_argument("--eval_batch_size", type=int,
                        help="batch size to use in evaluation dataset")
    parser.add_argument("--test_batch_size", type=int,
                        help="batch size to use in evaluation dataset")
    parser.add_argument("--max_epochs", type=int,
                        help="maximum number of trainin epochs (iterations)")
    parser.add_argument("--lr", type=float,
                        help="starting learning rate")
    parser.add_argument("--betas", type=tuple,
                        help="betas parameter for Adam optimizer")
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
    
    start_time = time.perf_counter()
    
    # datasets are already split into words/pre-tokenized (but not for GPT-2)
    print("Loading datasets")
    wiki_train = load_dataset(path=args.train_ds)
    wiki_val = load_dataset(path=args.val_ds)
    wiki_test = load_dataset(path=args.test_ds)
    
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # join the read in lines (they're separated by \n so joining with empty 
    # strings is fine)
    train_toks = tokenizer.tokenize("".join(wiki_train))
    val_toks = tokenizer.tokenize("".join(wiki_val))
    test_toks = tokenizer.tokenize("".join(wiki_test))
    
    # resize train set to selected nr of tokens
    train_toks = resize(toks=train_toks, n_tokens=args.train_set_size)
    
    # split into chunks of equal lengths, pad at the end and at the beginning
    # with eos
    eos = "|<endoftext>|"
    train_toks_chunked = list(chunk(train_toks, 1024, equal_len=True,
                           eos=eos, bos=eos))
    val_toks_chunked = list(chunk(val_toks, 1024, equal_len=True,
                         eos=eos, bos=eos))
    test_toks_chunked = list(chunk(test_toks, 1024, equal_len=True,
                         eos=eos, bos=eos))
    
    # create input indices
    train_input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in train_toks_chunked]
    val_input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in val_toks_chunked]
    test_input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in test_toks_chunked]
    
    # clear for memory (vars no longer needed)
    wiki_train, wiki_val, wiki_test = None, None, None
    train_toks, val_toks, test_toks = None, None, None
    train_toks_chunked, val_toks_chunked, test_toks_chunked = None, None, None
    
    # batch the data into new shape: 
    # shape = (batch_size, batch_id, sequence_length)
    train_ds = torch.tensor(data=batchify(np.asarray(train_input_ids), 
                                          bs=args.train_batch_size),
                            dtype=torch.long)

    val_ds = torch.tensor(data=batchify(np.asarray(val_input_ids), 
                                         bs=args.val_batch_size),
                           dtype=torch.long)
    
    # keep batch size 1 for testing
    test_ds = torch.tensor(data=batchify(np.asarray(test_input_ids), 
                                         bs=args.test_batch_size),
                           dtype=torch.long)
    
    end_time = time.perf_counter()
    
    # print some timing feedback
    elapsed = str(timedelta(seconds=round(end_time-start_time)))
    print("Dataset preparation took {} (HH:MM:SS)".format(elapsed))
    
    
    # ==== INITIALIZE MODEL, CONFIG AND EXPERIMENT CLASSES ===== #
    
    # initialize model with default config and move to device
    torch.manual_seed(args.seed)
    model = GPT2LMHeadModel(GPT2Config()).to(device)
    
    # declare optimizer
    optimizer=AdamW(params=model.parameters(), 
                    lr=args.lr, 
                    betas=args.betas)
    
    # initilize learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=args.max_epochs,
                                                num_cycles=0.5,
                                                last_epoch=-1)
    
    experiment = Experiment(model=model,
                            train_ds=train_ds,
                            eval_ds=val_ds,
                            test_ds=test_ds,
                            train_config=TrainConfig(bs=args.batch_size,
                                                     n_epochs=args.n_epochs),
                            eval_config=EvalConfig(bs=args.batch_size,
                                                   patience=args.es_patience),
                            optimizer=optimizer,
                            lr_scheduler=scheduler,
                            model_name=args.model_name,
                            savedir=args.savedir,
                            logdir=args.logdir)
    
    
    # ===== RUN EXPERIMENT =====#
    
    experiment.start(do_test=True)
    experiment.saveconfigs()

if __name__ == "__main__":

    runtime_code()
