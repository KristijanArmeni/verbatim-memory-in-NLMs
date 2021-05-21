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
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, top_k_top_p_filtering
import torch
from torch.nn import functional as F
from tqdm import trange
from typing import List
from string import punctuation
from tqdm import tqdm

# own modules
if "win" in sys.platform:
    sys.path.append(os.path.join(os.environ['homepath'], 'project', 'lm-mem', 'src'))
elif "linux" in sys.platform:
    sys.path.append(os.path.join(os.environ['HOME'], 'code', 'lm-mem', 'data'))
    

# ===== WRAPPERS FOR DATASET CONSTRUCTION ===== #

def mark_subtoken_splits(tokens):
    """
    function to keep track of whether or not a token was subplit into
    """
    ids = []
    count = 0
    for i in range(len(tokens)):
        
        # if the token is split, it does not gave the symbol for whitespace
        # if the word is at position 0, it also has to start a new token, so 
        # count it
        if "Ġ" in tokens[i]:
            count += 1
        
        if tokens[i] in punctuation:
            ids.append(-1)
        elif tokens[i] == "<|endoftext|>":
            ids.append(-2)
        else:
            ids.append(count)
    
    return ids

def assign_subtokens_to_groups(subtoken_splits, markers, ngram1_size, ngram2_size, n_repet):
    
    # this assumes both targets and interleaved groups are repated 5 times
    # we correct for this below
    codes = list(np.tile(np.repeat(markers, [ngram1_size, ngram2_size]), n_repet))
    
    # drop the last ngram as ngram2 is only repeated 5-1 times
    if ngram2_size != 0:
        del codes[-ngram2_size:]
    
    n_toks_no_punct = np.unique(subtoken_splits)[np.unique(subtoken_splits)>0]
    assert len(n_toks_no_punct) == len(codes)
    
    out = subtoken_splits.copy()
    
    punct_and_eos_codes = [-1, -2] # we leave these as they are
    for i, el in enumerate(subtoken_splits):
        
        if el not in punct_and_eos_codes:
            out[i] = codes[subtoken_splits[i]-1]

    return out

def concat_and_tokenize_inputs(prefixes=None, prompts=None, word_list1=None, word_list2=None,
                               ngram_size=None, tokenizer=None):
        
    """
    function that concatenates and tokenizes
    """
        
    metadata = {
        "stimid": [],
        "trialID": [],
        "positionID": [],
        "subtok": [],
        "list_len": [],
        "prefix": [],
        "prompt": [],
        }
    
    # join list elements into strings for tokenizer below
    input_seqs = [" " + ", ".join(tks) + "." for tks in word_list1]
    input_seqs2 = [" " + ", ".join(tks) + "." for tks in word_list2]
    
    # list storing outputs
    input_seqs_tokenized = []
    
    # loop over different prefixes:
    for prefix_key in prefixes.keys():

        # loop over prompts
        for prompt_key in prompts.keys():

            # loop over trials
            for i in range(len(input_seqs)):
                
                # tokenize strings separately to be able to construct markers for prefix, word lists etc.
                i1 = tokenizer.encode("<|endoftext|> " + prefixes[prefix_key], return_tensors="pt")   # prefix IDs, add eos token
                i2 = tokenizer.encode(input_seqs[i], return_tensors="pt") 
                i3 = tokenizer.encode(" " + prompts[prompt_key], return_tensors="pt")                       # prompt IDs
                i4 = tokenizer.encode(input_seqs2[i] + "<|endoftext|>", return_tensors="pt")

                # compose the input ids tensors
                input_ids = torch.cat((i1, i2, i3, i4), dim=1)
                
                input_seqs_tokenized.append(input_ids)

                # construct IDs for prefix, word lists and individual tokens
                # useful for data vizualization etc.
                trials = []
                positions = []
                split_ids = []
                
                for j, ids in enumerate((i1, i2, i3, i4)):
                    tmp = np.zeros(shape=ids.shape[1], dtype=int)  # code the trial structure
                    tmp[:] = j
                    tmp2 = np.arange(ids.shape[1])                 # create token position index
                    trials.append(tmp)
                    positions.append(tmp2)
                    split_ids.append(mark_subtoken_splits(tokenizer.convert_ids_to_tokens(ids[0])))
                
                metadata["trialID"].append(np.concatenate(trials).tolist())
                metadata["stimid"].append(i)
                metadata["positionID"].append(np.concatenate(positions).tolist())
                metadata["subtok"].append(np.concatenate(split_ids).tolist())
                metadata["list_len"].append(ngram_size)
                metadata["prefix"].append(prefix_key)
                metadata["prompt"].append(prompt_key)
                
    return input_seqs_tokenized, metadata


def concat_and_tokenize_inputs2(input_sets=None, ngram_size=None,
                                tokenizer=None):
        
    """
    function that concatenates and tokenizes
    """
        
    metadata = {
        "stimid": [],
        "trialID": [],
        "positionID": [],
        "subtok": [],
        "subtok_markers": [],
        "list_len": [],
        "prefix": [],
        "prompt": [],
        }
    
    input_seqs_tokenized = []

    # loop over distractor sets
    for dst_size in input_sets.keys():
        
        # get input lists interleaved with distractors of dst_size
        input_lists = input_sets[dst_size]
        
        # loop each input sequence
        for i in range(len(input_lists)):
            
            # tokenize strings separately to be able to construct markers for prefix, word lists etc.
            #i1 = tokenizer.encode("<|endoftext|> " + prefixes[prefix_key], return_tensors="pt")   # prefix IDs, add eos token
            input_ids = tokenizer.encode("<|endoftext|>" + input_lists[i] + "<|endoftext|>", return_tensors="pt") 

            # compose the input ids tensors
            #input_ids = torch.cat((i1, i2), dim=1)
            
            input_seqs_tokenized.append(input_ids)

            # construct IDs for prefix, word lists and individual tokens
            # useful for data vizualization etc.

            split_ids = mark_subtoken_splits(tokenizer.convert_ids_to_tokens(input_ids[0]))
            
            markers, ngram1_size, ngram2_size, n_repet = [1, 2], ngram_size, int(dst_size.strip("n")), 5
            
            split_ids_markers = assign_subtokens_to_groups(subtoken_splits=split_ids, 
                                                           markers=markers, 
                                                           ngram1_size=ngram1_size, 
                                                           ngram2_size=ngram2_size,
                                                           n_repet=n_repet)
            
            metadata["trialID"].append(np.ones(shape=input_ids.shape[1], dtype=int).tolist())
            metadata["stimid"].append(i)
            metadata["positionID"].append(np.arange(input_ids.shape[1]).tolist())
            metadata["subtok"].append(split_ids)
            metadata["subtok_markers"].append(split_ids_markers)
            metadata["list_len"].append(ngram_size)
            metadata["prompt"].append(dst_size.strip("n"))
                
    return input_seqs_tokenized, metadata


def interleave(items1, items2):
    
    items2.append("") # add a dummy element at the end
    return [val for pair in zip(items1, items2) for val in pair]


def interleave_targets_and_distractors(word_list, distractors):
    
    # conde the non-interleaved condition as well
    distractor_sizes = ["n0"] + list(distractors.keys()) 

    out = {key: [] for key in distractor_sizes}
    
    for dst_size in distractor_sizes:
        
        distractor_list = [None]
        
        if dst_size != "n0":
            distractor_list = distractors[dst_size]
        
        # loop over ngram chunks for each a trial
        for targets in word_list:
            
            for dst in distractor_list:
                
                nouns = targets
                
                # if there are distractors, interleave them
                if dst is not None:
                    nouns = interleave(items1=targets, items2=dst)
                
                trial = ", ".join([", ".join(e) for e in filter(None, nouns)]) + "."
                
                out[dst_size].append(" " + trial)

    return out


def sample_indices_by_group(groups, seed):
    
    """
    randomized_indices = sample_indices_by_group(groups, seed)
    
    input args:
        groups = np.array, array defining group membership (e.g. [0, 0, 0, 1, 1, 1])
        seed   = int, argument for np.random.RandomState
    output args:
        randomized_indices = np.array, randomly sampled indices of groups.size
    
    Helper function that creates randomized indices form np.arange(groups.size)
    by following the structure of group elements in group. It ensures that every
    element groups is paired with an element outside its own group.
    """
    
    out_ids = np.zeros(groups.shape, dtype=int)
    indices = np.arange(groups.size)
    rng = np.random.RandomState(seed)
    ignore_id = -1
    
    # number of selected samples must mach size of one group
    sample_size = np.sum(groups == 0)
    
    for group in np.unique(groups):
        
        # choose indices not from current group and not the ones already sampled
        candidate_pool = indices[(groups != group) & (indices != ignore_id)]
        
        sel_ids = rng.choice(a=candidate_pool, size = sample_size)
        out_ids[groups == group] = sel_ids
        
        # mark already selected indices
        indices[sel_ids] = ignore_id
        
    return out_ids

def ensure_list2_notequal(list1, list2, start_seed, seed_increment):
    
    """
    new_list2 = ensure_list2_notequal(list1, list2, start_seed, seed_increment)
    
    Helper function that ensures that all elements in list2 are not
    equal to elements in list1 by iteratively applying new perumtations
    with a new start_seed, incremented by seed_increment.
    """
    
    are_equal = [[t1 == t2] for t1, t2 in zip(list1, list2)]
    
    seed = start_seed
    
    # if lists are already disjoint, just return list2
    if ~np.any(are_equal):
        list2_new = list2
    
    else:
        # do this until elements of list1 and list2 are not equal
        while np.any(are_equal):
            
            rng = np.random.RandomState(seed)
            
            # create new permutations for l2 that are still equal
            list2_new = [rng.permutation(l2).tolist() if l1 == l2 else l2 
                         for l1, l2 in zip(list1, list2)]
            
            # update the criterion condition (none should be equal)
            are_equal = [[l1 == l2] for l1, l2 in zip(list1, list2_new)]
            
            # update seed for a new try
            seed += seed_increment
    
    return list2_new

# ===== EXPERIMENT CLASS ===== #

class Experiment(object):
    
    """
    Exp() class contains wrapper methods to run experiments with transformer models.
    """
    
    def __init__(self, model, tokenizer, context_len, device):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
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

    # loop over input list
    def ppl(self, input_ids, context_len, stride):
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
            sel_input_ids = input_ids[:, begin_loc: end_loc].to(self.device)

            # define target labels, use input ids as target outputs
            target_ids = sel_input_ids.clone()
            
            # do not compute the loss on  tokens (-100) that are used for context
            target_ids[:, :-trg_len] = -100  


            # set model to evaluation mode
            self.model.eval()
            
            # get model output
            with torch.no_grad():

               # compute neg log likelihood over target ids (n+1 in our case)
               # indices are shifted under the hood by model.__call__()
               outputs = self.model(sel_input_ids, labels=target_ids)
               
               # first element of the tuple contains the loss
               log_likelihood = outputs.loss.item() * trg_len  # not sure about this multiplication here (undoing averaging?)

               llh.append(log_likelihood)
               toks = self.tokenizer.decode(target_ids[0][-stride::])
               tokens.append(toks)  # store the last token (target_id)
               
        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / end_loc).cpu()
        return ppl, llh, tokens
    
    def start(self, input_sequences_ids) -> List:
        """
        experiment.start() will loop over prefixes, prompts, and word_lists and run the .ppl() method on them
        It returns a dict:
        outputs = {
            "sequence_ppl": [],
            "surp": [],
            "token": [],
            }
            
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
                                       stride=1)

            # store the output tuple and
            outputs["sequence_ppl"].append(ppl)
            outputs["surp"].append(surp)
            outputs["token"].append(toks)

        return outputs


def permute_attention_weights(model, seed):
    
    i=0
    # access transformer blocks
    for block in tqdm(model.transformer.h, desc="layer"):
        
        # make seed different for every layer
        rng = np.random.RandomState(seed+(5*i))
        
        # .weight is a rect matrix of stacked square matrices
        attn_dim = block.attn.c_attn.weight.shape[0]
        
        # spliting at dim 1 should result in 3 square matrices
        Q, K, V = block.attn.c_attn.weight.split(split_size=attn_dim, dim=1)
        
        w_shuf = []
        for w in (Q, K, V):
            w_shuf.append(torch.tensor(rng.permutation(w.detach().numpy())))
        
        new_qkv = torch.nn.Parameter(data=torch.cat(w_shuf, dim=1),
                                     requires_grad=False)
        
        block.attn.c_attn.weight = new_qkv
        
        i += 1
        
    return model

# ===== RUNTIME CODE WRAPPER ===== #

def runtime_code(): 
    
    
    from stimuli import prefixes, prompts
    
    
    # ===== INITIATIONS ===== #
    
    # collect input arguments
    parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")
    
    parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3"],
                        help="str, which scenario to use")
    parser.add_argument("--condition", type=str, choices=["repeat", "permute", "control"],
                        help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
    parser.add_argument("--paradigm", type=str, choices=["with-context", "repeated-ngrams"],
                        help="whether or not to permute the second word list")
    parser.add_argument("--context_len", type=int, default=1024,
                        help="length of context window in tokens for transformers")
    parser.add_argument("--model_type", type=str, default="pretrained", choices=["pretrained", "random", "random-att"],
                        help="whether or not to load a pretrained model or initialize randomly")
    parser.add_argument("--model_seed", type=int, default=12345,
                        help="seed value to be used in torch.manual_seed() prior to calling GPT2Model()")
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
    
    # manage platform dependent stuff
    if "win" in sys.platform: 
        data_dir = os.path.join(os.environ["homepath"], "project", "lm-mem", "src", "data")
    elif "linux" in sys.platform:
        data_dir = os.path.join(os.environ["HOME"], "code", "lm-mem", "data")

    print("condition == {}".format(argins.condition))
    print("scenario == {}".format(argins.scenario))
    
    # ===== DATA MANAGEMENT ===== #
    
    # load the word lists in .json files
    fname = os.path.join(data_dir, argins.input_filename)
    with open(fname) as f:
        
        print("Loading {} ...".format(fname))
        stim = json.load(f)
    
        # convert word lists to strings and permute the second one if needed
        # add space at the string onset
        word_lists1 = stim
    
    if argins.paradigm == "repeated-ngrams":
        
        fname = os.path.join(data_dir, "ngram-distractors.json")
        print("Loading {} ...".format(fname))
        with open(fname) as f:
            distractors = json.load(f)
    
    if argins.condition == "permute":
    
        # This condition test for the effect of word order
        # Lists have the same words, but the word order is permuted
        # int the second one
        word_lists2 = {key: [np.random.RandomState((543+j)*5).permutation(stim[key][j]).tolist()
                      for j in range(len(stim[key]))]
                      for key in stim.keys()}
        
        for list_size in word_lists2.keys():
            
            word_lists2[list_size] = ensure_list2_notequal(list1=word_lists1[list_size],
                                                           list2=word_lists2[list_size],
                                                           start_seed=123,
                                                           seed_increment=10)

       
        # make sure control tokens do not appear in the target lists
        for k in stim.keys():
            assert ~np.any([[t1 == t2] for t1, t2 in zip(word_lists1[k], word_lists2[k])])
    
    elif argins.condition == "control":
    
        # This serves as a control conditions
        # Here list length is the only common factor between two lists
        
        print("Creating control condition...")
    
        n_items_per_group = 10
        n_groups = len(word_lists1["n10"])//n_items_per_group
        groups = np.repeat(np.arange(0, n_groups), n_items_per_group)
        
        ids = sample_indices_by_group(groups=groups, seed=12345)
        
        word_lists2 = {key: np.asarray(stim[key])[ids].tolist() for key in stim.keys()}
        
        # make sure control tokens do not appear in the target lists
        for k in stim.keys():
            assert ~np.any([set(t1).issubset(set(t2)) 
                            for t1, t2 in zip(word_lists1[k], word_lists2[k])])
        
    else:
        
        word_lists2 = word_lists1
    
    # if n-gram experiment modify prompt and prefix dicts, recreate them on the fly
    # to only contain a single prompt
    if "ngram" in argins.input_filename and argins.paradigm == "with-context":
        # grab only the first prefix and prompt
        prefixes = {argins.scenario: {list(prefixes[argins.scenario].keys())[0] : list(prefixes[argins.scenario].values())[0]}}
        prompts = {argins.scenario: {list(prompts[argins.scenario].keys())[0] : list(prompts[argins.scenario].values())[0]}}   
    
    
    # ===== INITIATE EXPERIMENT CLASS ===== #
    
    # declare device and paths
    device = torch.device(argins.device if torch.cuda.is_available() else "cpu") 
        
    # setup the model
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # load pretrained small model
    if argins.model_type == "pretrained":
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # or initialize a random model
    elif argins.model_type == "random":
        # initialize with random weights
        torch.manual_seed(argins.model_seed)
        model = GPT2LMHeadModel(config=GPT2Config()) 
        
    # or initialize a random model
    elif argins.model_type == "random-att":
        
        # initialize with random weights
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        
        print("Permuting model attention weights ...")
        model = permute_attention_weights(model, seed=argins.model_seed)
        
    experiment = Experiment(model=model, tokenizer=tokenizer, 
                            context_len=argins.context_len,
                            device=device)
    
    # run the experiment for all possible word lists
    # construct input sequences
    
    # list storing output dataframes
    experiment_outputs = []
    
    for n_words in list(word_lists1.keys()):
        
        # ===== PREPARE INPUT SEQUENCES ===== #
        
        # inputs for contextualized paradigm
        if argins.paradigm == "with-context":
            
            # this routing loops over prompts and prefixes
            # it keeps track of that in meta_data
            input_sequences, meta_data = concat_and_tokenize_inputs(prompts=prompts[argins.scenario], 
                                                                    prefixes=prefixes[argins.scenario], 
                                                                    word_list1=word_lists1[n_words], 
                                                                    word_list2=word_lists2[n_words],
                                                                    ngram_size=n_words.strip("n"),
                                                                    tokenizer=tokenizer)
        
        # prepare input sequences for the repeated ngrams paradigm
        elif argins.paradigm == "repeated-ngrams":

            word_lists = interleave_targets_and_distractors(word_lists1[n_words], distractors)
            
                        
            input_sequences, meta_data = concat_and_tokenize_inputs2(input_sets=word_lists,
                                                                     ngram_size=int(n_words.strip("n")),
                                                                     tokenizer=tokenizer)
            
        
        # ===== RUN EXPERIMENT LOOP ===== #
        
        output_dict = experiment.start(input_sequences)
        
        
        # ===== FORMAT AND SAVE OUTPUT ===== #
        
        # convert the output to dataframe
        if argins.paradigm == "with-context":
            
            meta_cols = ["trialID", "positionID", "subtok"]
            colnames = ["token"] + meta_cols + ["surp"]
            arrays = [output_dict["token"], 
                      meta_data["trialID"], 
                      meta_data["positionID"],
                      meta_data["subtok"],
                      output_dict["surp"]]
        
        # this paradigm one has one extra meta column
        elif argins.paradigm == "repeated-ngrams":
            
            meta_cols = ["trialID", "positionID", "subtok", "subtok_markers"]
            colnames = ["token"] + meta_cols + ["surp"]
            arrays = [output_dict["token"], 
                      meta_data["trialID"], 
                      meta_data["positionID"],
                      meta_data["subtok"],
                      meta_data["subtok_markers"],
                      output_dict["surp"]]
        
        counter = 1  # counter for trials
        n_sequences = len(output_dict["surp"])
        
        # make a new single dict with all rows for the df below
        dfrows = {key_arr[0]: key_arr[1] for i, key_arr in enumerate(zip(colnames, arrays))}
        
        # loop over trials
        for i in range(0, n_sequences):
            
            # a list of lists (row values) for this sequence
            row_values = [dfrows[key][i] for key in dfrows.keys()]
            
            # convert the last two elements of the tuple to an array
            dftmp = pd.DataFrame(np.asarray(row_values).T,
                                 columns=colnames)
        
            # now add the constant values for the current sequence rows
            dftmp["stimid"] = meta_data["stimid"][i]                # stimulus id
            dftmp['prompt'] = meta_data["prompt"][i]                # add a column of prompt labels
            dftmp["list_len"] = meta_data["list_len"][i]            # add list length
            dftmp['stimID'] = counter                               # track sequence id
            dftmp['second_list'] = argins.condition                 # log condition of the second list
            
            if argins.paradigm == "with-context":
                dftmp['prefix'] = meta_data["prefix"][i]                # add a column of prefix labels
            
            experiment_outputs.append(dftmp)
            counter += 1
    
    experiment.model.to("cpu")
    
    # put into a single df and save
    output = pd.concat(experiment_outputs)
    
    # save output
    print("Saving {}".format(outpath))
    output.to_csv(outpath, sep=",")

# ===== RUN ===== #
if __name__ == "__main__":

    runtime_code()
