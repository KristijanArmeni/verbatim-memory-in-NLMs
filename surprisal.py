"""
surprisal.py is used to run the perplexity experiment with GPT-2
it relies on the Experiment() class which is just a class with wrapper methods
around the Transformers library.

Use as:

python experiment.py
or from an ipython console:
%run experiment.py ""

"""

import json
import sys, os
import argparse
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F
from tqdm import trange
from typing import List

# own modules
sys.path.append(os.path.join(os.environ['homepath'], 'code', 'lm-mem', 'data'))


# ===== WRAPPER FOR DATASET CONSTRUCTION ===== #

def concat_and_tokenize_inputs(prefixes, prompts, word_list1, word_list2,
                               tokenizer):
        
    """
    function that concatenates and tokenizes
    """
        
    metadata = {
        "trialID": [],
        "positionID": [],
        "list_len": [],
        "prefix": [],
        "prompt": [],
        }
    
    input_seqs_tokenized = []
    
    # loop over different prefixes:
    for prefix_key in prefixes.keys():

        # loop over prompts
        for prompt_key in prompts.keys():

            # loop over trials
            for i in range(len(word_list1)):

                # tokenize strings separately to be able to construct markers for prefix, word lists etc.
                i1 = tokenizer.encode("<|endoftext|> " + prefixes[prefix_key], return_tensors="pt")   # prefix IDs, add eos token
                i2 = tokenizer.encode(word_list1[i], return_tensors="pt")                             # word list IDs
                i3 = tokenizer.encode(prompts[prompt_key], return_tensors="pt")                       # prompt IDs
                i4 = tokenizer.encode(word_list2[i] + "<|endoftext|>", return_tensors="pt")

                # compose the input ids tensors
                input_ids = torch.cat((i1, i2, i3, i4), dim=1)
                
                input_seqs_tokenized.append(input_ids)

                # construct IDs for prefix, word lists and individual tokens
                # useful for data vizualization etc.
                trials = []
                positions = []
                for j, ids in enumerate((i1, i2, i3, i4)):
                    tmp = np.zeros(shape=ids.shape[1], dtype=int)  # code the trial structure
                    tmp[:] = j
                    tmp2 = np.arange(ids.shape[1])                 # create token position index
                    trials.append(tmp)
                    positions.append(tmp2)
                
                metadata["trialID"].append(np.concatenate(trials).tolist())
                metadata["positionID"].append(np.concatenate(positions).tolist())
                metadata["list_len"].append(len([e.strip().replace(".", "") for e in word_list1[i].split(",")]))
                metadata["prefix"].append(prefix_key)
                metadata["prompt"].append(prompt_key)
                
    return input_seqs_tokenized, metadata

# ===== EXPERIMENT CLASS ===== #

class Experiment(object):
    
    """
    Exp() class contains wrapper methods to run experiments with transformer models.
    """
    
    def __init__(self, model, tokenizer, input_sequences, context_len, device):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.input_sequences = input_sequences
        self.context_len = context_len
        self.model.to(device)

    def get_logits(self, input_string):
        
        inp_ids = self.tokenizer.encode(input_string, return_tensors="pt")

        # take the output logits of the model
        with torch.no_grad():
            logits = self.model(inp_ids)[0]  # tuple of shape = (logits, layer and attention output)

        return logits
            
    def get_probs(self, logits, n_tokens=5, k=10^3, p=0.9):
        
        """get_probs(self, logtis, k, p) is a convenience fun wrapping around 
        top_k_top_p_filtering(logits, top_k, top_p), calling F.softmax() on the output and sampling from   there
        """
        
        # take only the top 10 logits
        filtered_logits = top_k_top_p_filtering(logits[:, -1, :], top_k=k, top_p=p)

        # apply the softmax to the truncated distribution
        probs = F.softmax(filtered_logits, dim=-1)

        # now sample n tokens
        # next_tokens = torch.multinomial(probs, num_samples=n_tokens, replacement=False)

        token_prob, token_ids = torch.topk(x=probs, dim=1, k=n_tokens, sorted=True)

        return token_prob, token_ids

    def uniform_att(self, model):

        # access transformer blocks
        for block in self.model.transformer.h:

            Q, K, V = block.attn.c_attn.weight

        return model

    # loop over input list
    def ppl(self, input_ids, context_len, stride, device):
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
            target_ids[:, :-trg_len] = -100  # ignore the tokens that were used for context when computing the loss

            # set model to evaluation mode
            self.model.eval()

            # get model output
            with torch.no_grad():

               # compute log likelihood
               outputs = self.model(sel_input_ids, labels=target_ids)  # this returns avg log likelihood over sequence
               log_likelihood = outputs[0] * trg_len  # not sure about this multiplication here (undoing averaging?)

               llh.append(log_likelihood.tolist())
               toks = self.tokenizer.decode(target_ids[0][-stride::])
               tokens.append(toks)  # store the last token (target_id)

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / end_loc)
        return ppl, llh, tokens
    
    def run_perplexity(self, input_sequences_ids) -> List:
        """
        experiment.run() will loop over prefixes, prompts, and word_lists and run the Sampler on them
        It returns a list of tuples.
        """
        
        # output dict
        outputs = {
            "sequence_ppl": [],
            "surp": [],
            "token": [],
            }

        # loop over trials (input sequences)
        for input_ids in input_sequences_ids:
            
            # this returns surprisal (neg log ll)
            ppl, surp, toks = self.ppl(input_ids=input_ids.to(self.device), 
                                       context_len=self.context_len, 
                                       stride=1,
                                       device=self.device)

            # store the output tuple and
            outputs["sequence_ppl"].append(ppl)
            outputs["surp"].append(surp)
            outputs["token"].append(toks)

        return outputs

# ===== RUNTIME CODE WRAPPER ===== #

def runtime_code(): 
    
    from stimuli import prefixes, prompts
    
    
    # ===== INITIATIONS ===== #
    # collect input arguments
    parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")
    
    parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3"],
                        help="str, which scenario to use")
    parser.add_argument("--condition", type=str,
                        help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
    parser.add_argument("--context_len", type=int, default=1024,
                        help="length of context window in tokens for transformers")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        help="whether to run on cpu or cuda")
    parser.add_argument("--input_filename", type=str,
                        help="str, the name of the .json file containing word lists")
    parser.add_argument("--output_dir", type=str,
                        help="str, the name of folder to write the output_filename in")
    parser.add_argument("--output_filename", type=str,
                        help="str, the name of the output file saving the dataframe")
    
    argins = parser.parse_args()
    
    # construct output file name and check that it existst
    savedir = os.path.join(".", argins.output_dir)
    assert os.path.isdir(savedir)                                 # check that the folder exists
    outpath = os.path.join(savedir, argins.output_filename)
    
    print("condition == {}".format(argins.condition))
    print("scenario == {}".format(argins.scenario))
    
    
    # ===== DATASET MANAGEMENT ===== #
    
    # load the word lists in .json files
    with open(argins.input_filename) as f:
        print("Loading {} ...".format(argins.input_filename))
        stim = json.load(f)
    
    # convert word lists to strings and permute the second one if needed
    # add space at the string onset
    word_list1 = [" " + ", ".join(l) + "." for l in stim]
    if argins.condition == "permute":
    
        # This condition test for the effect of word order
        # Lists have the same words, but the word order is permuted
        # int the second one
        word_list2 = [" " + ", ".join(np.random.RandomState((543+j)*5).permutation(stim[j]).tolist()) + "."
                      for j in range(len(stim))]
    
    elif argins.condition == "control":
    
        # This serves as a control conditions
        # Here list length is the only common factor between two lists
    
        print("Creating reverse control condition...")
        print("Assuming input list can be evenly split into 3 lists each of len(list)==20!")
        len3 = stim[0:20]
        len5 = stim[20:40]
        len10 = stim[40::]
        word_list2 = [" " + ", ".join(l) + "." for lst in (len3, len5, len10)
                                  for l in reversed(lst)]
    
    else:
        word_list2 = word_list1
    
    # if n-gram experiment modify prompt and prefix dicts, recreate them on the fly
    # to only contain a single prompt
    if "ngram" in argins.input_filename:
        # grab only the first prefix and prompt
        prefixes = {argins.scenario: {list(prefixes[argins.scenario].keys())[0] : list(prefixes[argins.scenario].values())[0]}}
        prompts = {argins.scenario: {list(prompts[argins.scenario].keys())[0] : list(prompts[argins.scenario].values())[0]}}   
    
    print(word_list1)
    
     # declare device and paths
    device = torch.device(argins.device if torch.cuda.is_available() else "cpu") 
        
    # setup the model
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    
    # construct input sequences
    input_sequences, meta_data = concat_and_tokenize_inputs(prompts=prompts[argins.scenario], 
                                                            prefixes=prefixes[argins.scenario], 
                                                            word_list1=word_list1, 
                                                            word_list2=word_list2,
                                                            tokenizer=tokenizer)
    
    # ===== INITIATE EXPERIMENT CLASS ===== #
    
    experiment = Experiment(model=model, tokenizer=tokenizer, 
                            input_sequences=input_sequences,
                            context_len=argins.context_len,
                            device=device)
    
    # run experiment
    output_dict = experiment.run_perplexity(input_sequences)
    
    # ===== FORMAT AND SAVE OUTPUT ===== #
    
    # convert the output to dataframe
    dfout = []
    counter = 1  # counter for trials
    
    n_sequences = len(output_dict["surp"])
    
    # loop over trials
    for i in range(0, n_sequences):
        
        # convert the last two elements of the tuple to an array
        dftmp = pd.DataFrame(np.asarray([output_dict["token"][i], 
                                         meta_data["trialID"][i], 
                                         meta_data["positionID"][i], 
                                         output_dict["surp"][i]]).T,
                             columns=["token", "trialID", "positionID", "surp"])
    
        dftmp["ispunct"] = dftmp.token.isin([".", ":", ","])     # create punctuation info column
        dftmp['prefix'] = meta_data["prefix"][i]               # add a column of prefix labels
        dftmp['prompt'] = meta_data["prompt"][i]               # add a column of prompt labels
        dftmp["list_len"] = meta_data["list_len"][i]               # add list length
        dftmp['stimID'] = counter
        dftmp['second_list'] = argins.condition                  # log condition of the second list
    
        dfout.append(dftmp)
        counter += 1
    
    # put into a single df and save
    dfout = pd.concat(dfout)
    
    # save output
    print("Saving {}".format(outpath))
    dfout.to_csv(outpath, sep=",")

# ===== RUN ===== #
if __name__ == "__main__":

    runtime_code()