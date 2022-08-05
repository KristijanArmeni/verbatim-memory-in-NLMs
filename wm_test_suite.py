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
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, \
                         BertForMaskedLM, TransfoXLLMHeadModel,  TransfoXLTokenizer, \
                         AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F
from tqdm import trange
from typing import List
from string import punctuation
from tqdm import tqdm
import logging

from transformers.utils.dummy_tokenizers_objects import HerbertTokenizerFast

logging.basicConfig(format=("[INFO] %(message)s"), level=logging.INFO)

# own modules
if "win" in sys.platform:
    sys.path.append(os.path.join(os.environ['homepath'], 'project', 'lm-mem', 'src'))
elif "linux" in sys.platform:
    sys.path.append(os.path.join(os.environ['HOME'], 'project', 'lm-mem', 'src'))


# ===== WRAPPERS FOR DATASET CONSTRUCTION ===== #

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

    def __init__(self, model, ismlm, tokenizer, context_len, device):

        self.model = model
        self.ismlm = ismlm
        self.device = device
        self.tokenizer = tokenizer
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
            sel_input_ids = input_ids[:, begin_loc: end_loc].clone().to(self.device)

            # mask the final token in sequence if we're testing bert, so that we get prediction only for that token
            if self.ismlm:
                sel_input_ids[0, -1] = self.tokenizer.convert_tokens_to_ids("[MASK]")

            # use unmasked tokens as targets for loss if provided
            target_ids = input_ids[:, begin_loc: end_loc].to(self.device).clone()

            # do not compute the loss on  tokens (-100) that are used for context
            target_ids[:, :-trg_len] = -100

            if i in range(0, 5):
                print(f"inputs: {sel_input_ids}")
                print(f"targets: {target_ids}")

            # set model to evaluation mode
            self.model.eval()

            # get model output
            with torch.no_grad():

               # compute neg log likelihood over target ids (n+1 in our case)
               # indices are shifted under the hood by model.__call__()
               outputs = self.model(sel_input_ids, labels=target_ids)
               
               log_likelihood = outputs.loss.item() * trg_len  # not sure about this multiplication here (undoing averaging?)

               llh.append(log_likelihood)
               toks = self.tokenizer.decode(target_ids[0][-stride::])
               tokens.append(toks)  # store the last token (target_id)

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / end_loc).cpu()
        return ppl, llh, tokens

    def start(self, input_sequences_ids=None) -> List:
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
        for i, input_ids in enumerate(input_sequences_ids):

            # this returns surprisal (neg log ll)
            ppl, surp, toks = self.ppl(input_ids=input_ids.to(self.device),
                                       context_len=self.context_len,
                                       stride=1)

            # store the output tuple and
            outputs["sequence_ppl"].append(ppl)
            outputs["surp"].append(surp)
            outputs["token"].append(toks)

        return outputs


def permute_qk_weights(model=None, per_head=False, seed=None):

    i=0

    if per_head:
        print("shuffling within attn heads...")
    else:
        print("shuffling across attn heads...")

    # access transformer blocks
    for block in tqdm(model.transformer.h, desc="layer"):

        # make seed different for every layer
        rng = np.random.RandomState(seed+(5*i))

        # .weight is a rect matrix of stacked square matrices
        attn_dim = block.attn.c_attn.weight.shape[0]

        # spliting at dim 1 should result in 3 square matrices
        Q, K, V = block.attn.c_attn.weight.split(split_size=attn_dim, dim=1)

        # get the size of each head by diving the embedding size with
        # the number of layers
        head_size = model.config.n_embd//model.config.n_layer

        qk_shuf = []
        for w in (Q, K):

            if not per_head:

                s = w.shape # store original shape

                #flatten, permute across rows/cols and reshape back
                wtmp = rng.permutation(w.detach().numpy().flatten()).reshape(s)
                qk_shuf.append(torch.tensor(wtmp))

            elif per_head:

                # split attn weights into n_layer x n_head square matrices
                heads_shuf = []

                w_attn_heads = w.split(split_size=head_size, dim=1)

                # permute weights within each head
                for j, head in enumerate(w_attn_heads):

                    # pick different seed for layer and each head
                    rng = np.random.RandomState(seed+(5*i+j))
                    s = head.shape # store original shape

                    # flatten, permute across cols/rows, then reshape
                    wtmp = rng.permutation(head.detach().numpy().flatten()).reshape(s)

                    heads_shuf.append(torch.tensor(wtmp))

                qk_shuf.append(torch.cat(heads_shuf, dim=1))

        new_qkv = torch.nn.Parameter(data=torch.cat(qk_shuf + [V], dim=1),
                                     requires_grad=False)

        block.attn.c_attn.weight = new_qkv

        i += 1

    return model


# ===== Setup for gpt2 ====== #
def setup():
    logging.info("SETUP: nltk punkt")
    import nltk
    nltk.download("punkt")

    logging.info("SETUP: GPT2 Tokenizer")
    GPT2TokenizerFast.from_pretrained("gpt2")

    logging.info("SETUP: Head Model")
    GPT2LMHeadModel.from_pretrained("gpt2")

    return 0

# ===== helper function for development ===== #
def get_argins_for_dev(setup=False, 
                       inputs_file=None, 
                       inputs_file_unmasked=None,
                       inputs_file_info=None,
                       context_len=1024, 
                       checkpoint="gpt2",
                       tokenizer=None,
                       model_type=None,
                       model_seed=12345,
                       device="cuda",
                       ):

    argins = {

        "setup": setup,
        "inputs_file": inputs_file,
        "inputs_file_info": inputs_file_info,
        "inputs_file_unmasked": inputs_file_unmasked,
        "context_len": context_len,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "model_type": model_type,
        "model_seed": model_seed,
        "device": device

    }

    return argins

# ===== RUNTIME CODE WRAPPER ===== #
def runtime_code():

    # ===== INITIATIONS ===== #
    from types import SimpleNamespace

    # collect input arguments
    parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")

    parser.add_argument("--setup", action="store_true",
                        help="downloads and places nltk model and Tokenizer")
    parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
                        help="str, which scenario to use")
    parser.add_argument("--condition", type=str)
    parser.add_argument("--inputs_file", type=str, help="json file with input sequence IDs which are converted to tensors")
    parser.add_argument("--inputs_file_unmasked", type=str, help="json file with input sequence IDs which are not masked", default="")
    parser.add_argument("--inputs_file_info", type=str, help="json file with information about input sequences")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--context_len", type=int, default=1024,
                        help="length of context window in tokens for transformers")
    parser.add_argument("--model_type", type=str,
                        help="model label controlling which checkpoint to load")
    # To download a different model look at https://huggingface.co/models?filter=gpt2
    parser.add_argument("--checkpoint", type=str, default="gpt2",
                        help="the path to folder with pretrained models (expected to work with model.from_pretraiend() method)")
    parser.add_argument("--model_seed", type=int, default=12345,
                        help="seed value to be used in torch.manual_seed() prior to calling GPT2Model()")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        help="whether to run on cpu or cuda")
    parser.add_argument("--output_dir", type=str,
                        help="str, the name of folder to write the output_filename in")
    parser.add_argument("--output_filename", type=str,
                        help="str, the name of the output file saving the dataframe")

    argins = parser.parse_args()

    argins = SimpleNamespace(**get_argins_for_dev(inputs_file="/home/ka2773/project/lm-mem/src/data/transformer_input_files/transfo-xl_repeat_sce1_5.json",
                                                  inputs_file_unmasked="",
                                                  inputs_file_info="/home/ka2773/project/lm-mem/src/data/transformer_input_files/transfo-xl_repeat_sce1_5_info.json",
                                                  checkpoint="transfo-xl-wt103",
                                                  tokenizer="transfo-xl-wt103"))

    if argins.setup:
        setup()

    sys.path.append("/home/ka2773/project/lm-mem/src/data/")
    
    # ===== LOAD INPUTS ===== #
    with open(argins.inputs_file, "r") as fh:
        input_sequences = [torch.tensor(e) for e in json.load(fh)]

    input_sequences_unmasked = None
    if argins.inputs_file_unmasked != "":
        with open(argins.inputs_file_unmasked, "r") as fh:
            input_sequences_unmasked = [torch.tensor(e) for e in json.load(fh)]

    with open(argins.inputs_file_info, "r") as fh:
        input_sequences_info = json.load(fh)

    # ===== INITIATE EXPERIMENT CLASS ===== #

    # declare device and paths
    device = torch.device(argins.device if torch.cuda.is_available() else "cpu")

    # pretrained models
    logging.info("Using {} model".format(argins.model_type))
    logging.info("Loading checkpoint {}".format(argins.checkpoint))

    ismlm = False
    if argins.checkpoint == "bert-base-uncased":
        model = BertForMaskedLM.from_pretrained(argins.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(argins.tokenizer)
        ismlm = True

    elif argins.checkpoint == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    elif argins.checkpoint == "transfo-xl-wt103":
        model = TransfoXLLMHeadModel.from_pretrained(argins.checkpoint)
        tokenizer = TransfoXLTokenizer.from_pretrained(argins.tokenizer)

    # or initialize a random model
    if argins.model_type == "random":
        # initialize with random weights
        torch.manual_seed(argins.model_seed)
        model = GPT2LMHeadModel(config=GPT2Config())

    # permute the weights of gpt-small
    elif argins.model_type == "random-att":

        logging.info("Permuting model attention weights ...\n")
        model = permute_qk_weights(model, per_head=False,
                                   seed=argins.model_seed)

    # permute attenion heads of gpt2 small
    elif argins.model_type == "random-att-per-head":

        logging.info("Permuting Q and K weights ...\n")
        model = permute_qk_weights(model, per_head=True,
                                   seed=argins.model_seed)

    # shuffle embedding vectors of gpt2 small
    elif argins.model_type == "shuff-wpe":

        logging.info("Permuting token positions in wpe...")
        rng = np.random.RandomState(seed=argins.model_seed)

        wpe = model.transformer.wpe.weight # shape = (token_positions, embedding_dim)

        # permutation only permutes across 0 dim (rows=token_positions)
        wpe_shuf = torch.tensor(rng.permutation(wpe.detach().numpy()))
        model.transformer.wpe.weight = torch.nn.Parameter(data=wpe_shuf,
                                                          requires_grad=False)

    # set to evaluation mode
    model.eval()

    #initialize experiment class
    experiment = Experiment(model=model, ismlm=ismlm,
                            tokenizer=tokenizer,
                            context_len=argins.context_len,
                            device=device)

    # run the experiment for all possible word lists
    # construct input sequences

    # list storing output dataframes
    experiment_outputs = []

    # ===== RUN EXPERIMENT LOOP ===== #

    output_dict = experiment.start(input_sequences_ids = input_sequences)


    # ===== FORMAT AND SAVE OUTPUT ===== #

    meta_cols = ["trialID", "positionID", "subtok"]
    colnames = ["token"] + meta_cols + ["surp"]
    arrays = [output_dict["token"],
                input_sequences_info["trialID"],
                input_sequences_info["positionID"],
                input_sequences_info["subtok"],
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
        dftmp["stimid"] = input_sequences_info["stimid"][i]                # stimulus id
        dftmp['prompt'] = input_sequences_info["prompt"][i]                # add a column of prompt labels
        dftmp["list_len"] = input_sequences_info["list_len"][i]            # add list length
        dftmp['stimID'] = counter                               # track sequence id
        dftmp['second_list'] = argins.condition                 # log condition of the second list

        experiment_outputs.append(dftmp)
        counter += 1

    experiment.model.to("cpu")

    # put into a single df and save
    output = pd.concat(experiment_outputs)

    # save output
    outpath = os.path.join(argins.output_dir, argins.output_filename)
    print("Saving {}".format(os.path.join(outpath)))
    output.to_csv(outpath, sep=",")

# ===== RUN ===== #
if __name__ == "__main__":

    runtime_code()
