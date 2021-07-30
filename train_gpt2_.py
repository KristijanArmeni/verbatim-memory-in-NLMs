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
from datetime import timedelta
from tqdm import tqdm, trange
import logging
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel, \
                        Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
                        EarlyStoppingCallback

# own module
from dataset import WikiTextDataset

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


def compute_perplexity(model, input_ids, tokenizer, context_len, stride, device):
        """
        method for computing token-by-token negative log likelihood on input_ids
        taken from: https://huggingface.co/transformers/perplexity.html
        """

        llh = []  # variable storing token-by-token neg ll
        tokens = []   # variable storing token strings to have along with -ll in the output
        
        # loop over tokens in input sequence
        for i in trange(0, input_ids.size(1), stride, desc="Computing perplexity: "):

            # define the start and endpoints of input indices for current loop
            begin_loc = max(i + stride - context_len, 0)  # define the non-negative onset location
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # select the current input index span
            sel_input_ids = input_ids[:, begin_loc: end_loc].to(device)

            # define target labels, use input ids as target outputs
            target_ids = sel_input_ids.clone()
            
            # do not compute the loss on  tokens (-100) that are used for context
            target_ids[:, :-trg_len] = -100  


            # set model to evaluation mode
            model.eval()
            
            # get model output
            with torch.no_grad():

               # compute neg log likelihood over target ids (n+1 in our case)
               # indices are shifted under the hood by model.__call__()
               outputs = model(sel_input_ids, labels=target_ids)
               
               # first element of the tuple contains the loss
               log_likelihood = outputs.loss.item() * trg_len  # not sure about this multiplication here (undoing averaging?)

               llh.append(log_likelihood)
                
               # only output tokens if we are computing token-by-token ppl
               # (i.e. stride == 1)
               if stride == 1:
                  toks = tokenizer.decode(target_ids[0][-stride::])
                  tokens.append(toks)  # store the last token (target_id)

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / end_loc).cpu()
        return ppl, llh, tokens

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
    parser.add_argument("--tokenizer_path", type=str,
                        help="path to mergex.txt and vocab.json files")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="device used for training")
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_test", action='store_true', default=True)

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
    parser.add_argument("--embed_dim", type=int, default=100,
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
    
    # training regime input args (input to TrainingArguments() class)
    parser.add_argument("--lr", type=float,
                        help="starting learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="the scheduler for learning rate changes during learning")
    parser.add_argument("--adam_beta1", type=float, default=0.5,
                        help="beta1 parameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.05,
                        help="beta2 parameter for Adam optimizer")
    parser.add_argument("--num_lr_warmup_steps", type=int,
                        help="number of consecutive epochs for which learning" 
                        "rate is increased linearly")
    parser.add_argument("--num_eval_steps", type=int, 
                        help="number of steps after which evaluation is performed")
    parser.add_argument("--num_logging_steps", type=int,
                        help="number of steps after which to perform logging to wandb")
    parser.add_argument("--num_save_steps", type=int,
                        help="number of steps after which checkpoints are saved")
    parser.add_argument("--es_patience", type=int,
                        help="nr of consecutive epochs to wait for decreasing loss" 
                        "before stopping training")
    parser.add_argument("--es_delta", type=float, default=0.01,
                        help="the smallest change in loss between two evaluations that still counts as improvement, if c                        hange is less than this value, otherwise early stopping counter starts")

    # test arguments
    parser.add_argument("--test_stride", type=int, 
                        help="stride to use on ppl computation")

    # wandb params
    parser.add_argument("--wandb_key", type=str, help="authorization key to loging to wandb")
    parser.add_argument("--wandb_dir", type=str, help="directory where wandb files are written to")
    parser.add_argument("--wandb_notes", type=str, default="", help="notes for wandb logging")
    parser.add_argument("--wandb_name", type=str, default="run-name", help="run name for wandb logging")
    parser.add_argument("--wandb_project", type=str, help="project name for wandb logging")
    parser.add_argument("--wandb_group", type=str, help="label to group runs into")
    parser.add_argument("--wandb_tags", type=str, help="wandb tags to add to the run")
    parser.add_argument("--wandb_mode", type=str, choices=["disabled", "offline", "online"], 
                        help="control of wandb logging mode")
    parser.add_argument("--wandb_disabled", action='store_true', 
                        help="whether to turn wandb loggin off")

    # savedir params
    parser.add_argument("--savedir", type=str,
                        help="path where the model weights will be stored")
    parser.add_argument("--logdir", type=str,
                        help="path where the model weights will be stored")
    
    args = parser.parse_args()
    print(args)

    # use cuda if available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.device == "cuda":
        print_cuda_mem_info()
    
    # utility function from transformers (sets seed in torch and numpy)
    transformers.set_seed(args.seed)  
    
    # set logging verbosity output
    transformers.logging.set_verbosity_info()

    # load tokenizer trained in dataset.py
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    
    # load in retokenized files
    train_ds = WikiTextDataset(tokenizer=tokenizer)
    train_ds.make_input_sequences(json_path=args.train_ds,
                                  sequence_length=args.sequence_len)
    
    eval_ds = WikiTextDataset(tokenizer=tokenizer)
    eval_ds.make_input_sequences(json_path=args.val_ds,
                                 sequence_length=args.sequence_len)
    
    test_ds = WikiTextDataset(tokenizer=tokenizer)
    test_ds.make_input_sequences(json_path=args.test_ds,
                                 sequence_length=None)
    
    
    # set up some GPT2Config parameters
    # we keep n_positions and n_ctx equal 
    config = GPT2Config(n_positions=args.sequence_len,
                        n_ctx=args.sequence_len,
                        n_embd=args.embed_dim,
                        n_layer=args.n_layer,
                        n_head=args.n_head,
                        vocab_size=len(tokenizer.get_vocab()),
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id)
    

    # Training arguments
    train_args = TrainingArguments(
        output_dir=args.savedir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.lr,
        warmup_steps=args.num_lr_warmup_steps,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        logging_steps=args.num_logging_steps,
        eval_steps=args.num_eval_steps,
        save_steps=args.num_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        fp16=True,
        disable_tqdm=False,
        report_to="wandb",
        run_name=args.wandb_name,
        )
    
    # initialize data collator class
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)
    
    # add authorization token for wandb logging
    os.environ["WANDB_API_KEY"] = args.wandb_key
    
    # init wandb before Trainer, this will override the default callback in Trainer
    wandb.init(
        dir=args.wandb_dir if args.wandb_dir else None,
        project=args.wandb_project if args.wandb_project else None,
        name=args.wandb_name if args.wandb_name else None,
        tags=args.wandb_tags if args.wandb_tags else None,
        notes=args.wandb_notes if args.wandb_notes else None,
        group=args.wandb_group if args.wandb_group else None,
        mode=args.wandb_mode if args.wandb_mode else "online",
        )
   
    # initialize the model from configuration
    model = GPT2LMHeadModel(config=config)

    # initialize model and use data parallelism if multiple GPUs are avaiilable
    if torch.cuda.device_count() > 1:
        logging.info("Found {} GPUs, using nn.DataParallel".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # move to gpus
    model.to(device)

    # run training routine
    if args.do_train:
        
        # initialize trainer class and model form config
        trainer = Trainer(args=train_args,
                          model=model,
                          data_collator=data_collator,
                          train_dataset=train_ds,
                          eval_dataset=eval_ds,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience,
                                                           early_stopping_threshold=args.es_delta)]
                          )
    
        # hyper-param search here if needed
        # best_trial = trainer.hyperparameter_search(
        #                               direction='minimize',
        #                               search_alg='something')
        
        # call training routine
        trainer.train()
    
        # this must hold, so that we just refer to <model> name below
        assert id(model) == id(trainer.model)

    # compute and log test perplexity
    if args.do_test:
        
        ppl, _, _ = compute_perplexity(model=model,
                                       tokenizer=tokenizer,
                                       input_ids=torch.tensor([test_ds.x]),
                                       context_len=args.sequence_len,
                                       stride=args.test_stride,
                                       device=device)
        
        logging.info("Test ppl: {}".format(ppl))

        # log some info from testing
        wandb.log(
            data={'test-ppl': ppl, 
                  'test-stride': args.test_stride, 
                  'test-context-len':args.sequence_len}
            )
    
    # end logging
    wandb.finish()

if __name__ == "__main__":

    runtime_code()
