# -*- coding: utf-8 -*-
"""
script for training GPT-2 on wikitext-103

Created on Thu Mar 18 16:03:16 2021

@author: karmeni1
"""

import argparse
import os
import time
from ast import literal_eval
import sys
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import json
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel, \
                        Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
                        logging

# ============================== #
# ===== DATASET MANAGEMENT ===== #
# ============================== #

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

# wrapper function
def print_cuda_mem_info(device_id=0):
    
    t = torch.cuda.get_device_properties(device_id).total_memory
    r = torch.cuda.memory_reserved(device_id)
    a = torch.cuda.memory_allocated(device_id)
    f = r-a
    
    gb_factor = 0.9313*10e-10
    
    print("total GPU mem (gb): {}".format(round(t*gb_factor)))
    print("reserved GPU mem (gb): {}".format(r*gb_factor))
    print("allocated GPU mem (gb): {}".format(a*gb_factor))
    print("available GPU mem (gb): {}".format(f*gb_factor))

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
                 device = None,
                 lr_scheduler=None,
                 model_name=None,
                 savedir=None,
                 logdir=None):
        
        self.x_train = x_train
        self.x_eval = x_eval
        self.x_test = x_test
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.cfg_train = train_config
        self.cfg_eval = eval_config
        
        self.log = {
            "train_loss": [],
            "batch_id": [],
            "eval_loss": [],
            "test_ppl": [],
            }
        
    def start(self, do_test=False):
        
        # initialize early stopping class 
        early_stopping = self.EarlyStopping(model=self.model,
                                            patience=self.cfg_eval.es_patience)
        
        print("Starting training...\n")
        
        n_epochs = self.cfg_train.max_epochs

        # loop over epochs
        for n in list(range(0, n_epochs)):
            
            print("Epoch {}/{} ({}%).".format(n, n_epochs, 
                                             (n/n_epochs)*100))
            
            start_time = time.perf_counter()

            # run .train and .evaluate routines
            train_loss = self.train(x=self.x_train.to(self.device))
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
            self.log["test_ppl"].append(np.exp(test_loss))

        
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
        summed_loss = 0
        
        # loop over training samples
        for batch_idx in tqdm(range(0, n_batches), desc="train set", unit="batch"):
            
            batch_loss = self.train_step(x_batch=x[:, batch_idx, :])
            summed_loss += batch_loss.numpy().item()
            
            
        # append average loss over batches for this epoch
        self.log["train_loss"].append(summed_loss/n_batches)
         
        return summed_loss/n_batches
    
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
        self.lr_scheduler.step()
        
        # clear the gradients
        self.model.zero_grad()
        
        return output.loss.detach().cpu()
        
    def evaluate(self, x):
        
        self.model.eval()
        
        with torch.no_grad():
            
            # assume batch id is in second dimension
            n_batches = x.shape[1]
            
            # set loss to 0 at the start of the epoch
            summed_loss = 0
            
            # loop over training samples
            for batch_idx in tqdm(range(0, n_batches), desc="validation set:",
                                  unit="batch"):
                
                x_batch = x[:, batch_idx, :]
                
                output = self.model(input_ids=x_batch, labels=x_batch)
                
                # first tuple element is loss
                batch_loss = output[0]
                
                summed_loss += batch_loss.cpu().numpy().item()
            
            # store average eval loss for this epoch
            self.log["eval_loss"].append(summed_loss/n_batches)
        
        return summed_loss/n_batches
    
    def save_checkpoint(self, savename=None):
        
        if savename is None:
            savename = os.path.join(self.savedir, self.model_name)
        
        print("Saving checkpoint as:\n{}".format(savename))
        
        torch.save(self.model.cpu().state_dict(), 
                   savename)

    def save_configs(self, full_filename):
        
        pass

    class EarlyStopping(object):
        
        """
        EarlyStopping() is a helpher class that keeps track of evaluation loss.
        EarlyStopping.check() halts Experiment().start() subroutine if evaluation 
        set loss has not decreased for n == EarlyStopping().patience epochs in a row.
        """
        
        def __init__(self, model=None, patience=None):
            
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
    
    def read_and_preprocess_txt(self, path, sequence_length):
            
    
        with open(path, "r", encoding="utf-8") as file:
            
            lines = file.readlines()
        
        # retokenize the entire test set
        print("Retokenizing {}".format(path))
        ds_retok = []
        for l in tqdm(lines, desc="line"):
            ds_retok += self.tokenizer.encode(l)
        
        # clear lines for memory
        lines = None
        
        # split list of input tokens into list of elements of size max_len
        print("Chunking samples in self.x to length of {}".format(sequence_length))
        self.x = list(self.chunk_list(ds_retok, 
                                      n=sequence_length,
                                      bos=self.tokenizer.bos_token_id,
                                      eos=self.tokenizer.eos_token_id))
        

    def __len__(self):
        
        return len(self.x)
    
    def __getitem__(self, index):
        
        return torch.tensor(self.x[index])

# ======================== #
# ===== RUNTIME CODE ===== #
# ======================== #

def runtime_code():
    
    import wandb
    
    # collect input arguments
    parser = argparse.ArgumentParser()
    
    # general input args
    parser.add_argument("--datadir", type=str,
                        help="path to dataset .json files")
    parser.add_argument("--train_ds", type=str,
                        help="path to training dataset")
    parser.add_argument("--val_ds", type=str,
                        help="path to validation set")
    parser.add_argument("--test_ds", type=str,
                        help="path to tokenized file")
    parser.add_argument("--model_name", type=str,
                        help="filename for saving model checkpoint to --savedir")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="device used for training")

    # GPTConfig arguments
    parser.add_argument("--seed", type=int,
                        help="seed used in torch.manual_seed() for reproducibility")
    parser.add_argument("--sequence_len", type=int, default=1024,
                        help="length of input sequence for the model")
    parser.add_argument("--n_ctx", type=int, default=1024,
                        help="length of context mask, passed to GPT2Config() class")
    parser.add_argument("--n_layer", type=int, default=12,
                        help="number of layers, passed to GPT2Config() class")
    parser.add_argument("--n_head", type=int, default=12,
                        help="number of attention heads, passed to GPT2Config() class")
    
    # training set arguments
    parser.add_argument("--train_batch_size", type=int,
                        help="batch size to use in training ddataset")
    parser.add_argument("--eval_batch_size", type=int,
                        help="batch size to use in evaluation dataset")
    parser.add_argument("--test_batch_size", type=int,
                        help="batch size to use in evaluation dataset")
    parser.add_argument("--max_epochs", type=int,
                        help="maximum number of trainin epochs (iterations)")
    parser.add_argument("--train_set_size", type=str, default="40",
                        help="size of the traininset (in million tokens)")
    
    # training regime input args
    parser.add_argument("--lr", type=float,
                        help="starting learning rate")
    parser.add_argument("--betas", type=str,
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
    
    if args.device == "cuda":
        
        print_cuda_mem_info()
    
    # own modules
    if "win" in sys.platform:
        datapath = os.path.join(os.environ['homepath'], 'project', 'lm-mem', 'data')
    elif "linux" in sys.platform:
        datapath = os.path.join(os.environ['HOME'], 'code', 'lm-mem', 'data')
    
    # set logging verbosity output
    logging.set_verbosity_info()
    
    #tokenizer = ByteLevelBPETokenizer()
    
    #tokenizer.train(files=[os.path.join(datapath, "wikitext-103", "wiki.train.tokens")],
    #                vocab_size=28439, 
    #                min_frequency=1, 
    #                special_tokens=["<|endoftext|>", "<pad>"])
    
    # Save files to disk
    #tokenizer.save_model(datapath, "wikitext-103")
    
    # load the pretrained tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join(datapath, "GPT2_wikitext103"), max_len=512)


    train_ds = WikiTextDataset(tokenizer=tokenizer)
    train_ds.read_and_preprocess_txt(path=os.path.join(datapath, "wikitext-103", args.test_ds),
                                     sequence_length=args.sequence_len)
    
    eval_ds = WikiTextDataset(tokenizer=tokenizer)
    eval_ds.read_and_preprocess_txt(path=os.path.join(datapath, "wikitext-103", args.val_ds),
                                    sequence_length=args.sequence_len)
    
    # initialize model with default config and move to device
    print("Loading model to {}".format(args.device))
    torch.manual_seed(args.seed)
    
    # set up some GPT2Config parametersqq
    # we keep n_positions and n_ctx equal 
    config = GPT2Config(n_positions=args.sequence_len,
                        n_ctx=args.sequence_len,
                        n_embed=100,
                        n_layer=args.n_layer,
                        n_head=args.n_head,
                        vocab_size=len(tokenizer.get_vocab()),
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id)
    
    # Training argumentns
    train_args = TrainingArguments(
        output_dir=args.savedir,
        num_train_epochs=args.max_epochs,
        adam_beta1=literal_eval(args.betas)[0],
        adam_beta2=literal_eval(args.betas)[1],
        learning_rate=args.lr,
        warmup_steps=args.num_lr_warmup_steps,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        disable_tqdm=False,
        report_to="wandb",
        run_name="test",
        )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)
    
    wandb.login()
    wandb.init(project="test")
    
    trainer = Trainer(args=train_args,
                      model=GPT2LMHeadModel(config=config).to(device),
                      data_collator=data_collator,
                      train_dataset=train_ds,
                      eval_dataset=eval_ds)
    
    trainer.train()


if __name__ == "__main__":

    runtime_code()
