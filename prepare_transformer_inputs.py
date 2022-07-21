import json
import sys, os
import argparse
import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast, AutoTokenizer
import torch
from torch.nn import functional as F
from tqdm import trange
from typing import List
from string import punctuation
from tqdm import tqdm
import logging

logging.basicConfig(format=("[INFO] %(message)s"), level=logging.INFO)

# own modules
if "win" in sys.platform:
    sys.path.append(os.path.join(os.environ['homepath'], 'project', 'lm-mem', 'src'))
elif "linux" in sys.platform:
    sys.path.append(os.path.join(os.environ['HOME'], 'project', 'lm-mem', 'src'))


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

def concat_and_tokenize_inputs(prefix=None, prompt=None, word_list1=None, word_list2=None,
                               ngram_size=None, tokenizer=None, ismlm=False):

    """
    function that concatenates and tokenizes
    """

    metadata = {
        "stimid": [],
        "trialID": [],
        "positionID": [],
        "subtok": [],
        "list_len": [],
        }

    # join list elements into strings for tokenizer below
    input_seqs = [" " + ", ".join(tks) + "." for tks in word_list1]
    if ismlm:
        input_seqs2 = [" " + ", ".join(np.repeat("[MASK]", len(tks))) + "." for tks in word_list2]  # masked tokens
        input_seqs2_ = [" " + ", ".join(tks) + "." for tks in word_list2]                           # unmasked tokens
        eos1 = "[CLS]"
        eos2 = "[SEP]"
    else:
        input_seqs2 = [" " + ", ".join(tks) + "." for tks in word_list2]
        eos1 = "<|endoftext|>"
        eos2 = "<|endoftext|>"

    logging.info(f"Using {eos1} and {eos2} as start and end eos tokens, respectively")

    # list storing outputs
    input_seqs_tokenized = []
    input_seqs_tokenized_unmasked = []

    # loop over trials
    for i in trange(len(input_seqs), desc="sequence: "):

        # tokenize strings separately to be able to construct markers for prefix, word lists etc.
        i1 = tokenizer.encode(eos1 + " " + prefix, add_special_tokens=False, return_tensors="pt")   # prefix IDs, add eos token
        i2 = tokenizer.encode(input_seqs[i], add_special_tokens=False, return_tensors="pt")
        i3 = tokenizer.encode(" " + prompt, add_special_tokens=False, return_tensors="pt")                       # prompt IDs
        i4 = tokenizer.encode(input_seqs2[i] + eos2, add_special_tokens=False, return_tensors="pt")

        # compose the input ids tensors
        input_ids = torch.cat((i1, i2, i3, i4), dim=1)

        input_seqs_tokenized.append(input_ids)

        # tokenize unmasked tokens to be able to compute loss later on
        if ismlm:
            i4_ = tokenizer.encode(input_seqs2_[i] + eos2, add_special_tokens=False, return_tensors="pt")
            input_ids_unmasked = torch.cat((i1, i2, i3, i4_), dim=1)
            input_seqs_tokenized_unmasked.append(input_ids_unmasked)

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

    return input_seqs_tokenized, input_seqs_tokenized_unmasked, metadata


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

def get_args_for_dev(setup=False, scenario="sce1", condition="repeat", path_to_tokenizer="gpt2", 
                     device="cuda", input_filename="random_lists.json", output_dir=None, output_filename=None ):

    args = {
        "setup": False,
        "scenario": scenario,
        "condition": condition,
        "path_to_tokenizer": path_to_tokenizer,
        "device": device,
        "input_filename": "random_lists.json",
        "output_dir": output_dir,
        "output_filename": output_filename

    }

    return args

def main():

    from types import SimpleNamespace

    sys.path.append("/home/ka2773/project/lm-mem/src/data/")
    from stimuli import prefixes, prompts

    # ===== INITIATIONS ===== #

    # collect input arguments
    parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")

    parser.add_argument("--setup", action="store_true",
                        help="downloads and places nltk model and Tokenizer")
    parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
                        help="str, which scenario to use")
    parser.add_argument("--condition", type=str, choices=["repeat", "permute", "control"],
                        help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
    # To download a different model look at https://huggingface.co/models?filter=gpt2
    parser.add_argument("--path_to_tokenizer", type=str, default="gpt2",
                        help="the path to tokenizer folder (expected to work with tokenizer.from_pretrained() method")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        help="whether to run on cpu or cuda")
    parser.add_argument("--input_filename", type=str,
                        help="str, the name of the .json file containing word lists")
    parser.add_argument("--output_dir", type=str,
                        help="str, the name of folder to write the output_filename in")
    parser.add_argument("--output_filename", type=str,
                        help="str, the name of the output file saving the dataframe")

    argins = parser.parse_args()

    # if needed for development create the argins namespace with some default values
    # argins = SimpleNamespace(**get_args_for_dev(path_to_tokenizer="bert-base-uncased", 
    #                                            output_dir="data/transformer_input_files", 
    #                                            output_filename="bert-base-uncased"))

    # construct output file name and check that it existst
    savedir = os.path.join(".", argins.output_dir)
    assert os.path.isdir(savedir)                                 # check that the folder exists
    #outpath = os.path.join(savedir, argins.output_filename)

    # manage platform dependent stuff
    # if "win" in sys.platform:
    #     data_dir = os.path.join(os.environ["homepath"], "project", "lm-mem", "src", "data")
    # elif "linux" in sys.platform:
    #     data_dir = os.path.join(os.environ["HOME"], "code", "lm-mem", "data")
    data_dir = "./data"

    logging.info("condition == {}".format(argins.condition))
    logging.info("scenario == {}".format(argins.scenario))

    # ===== DATA MANAGEMENT ===== #

    # load the word lists in .json files
    fname = os.path.join(data_dir, argins.input_filename)
    with open(fname) as f:

        logging.info("Loading {} ...".format(fname))
        stim = json.load(f)

        # convert word lists to strings and permute the second one if needed
        # add space at the string onset
        word_lists1 = stim


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

        logging.info("Creating control condition...")

        n_items_per_group = 10
        n_groups = len(word_lists1["n10"])//n_items_per_group
        groups = np.repeat(np.arange(0, n_groups), n_items_per_group)

        ids = sample_indices_by_group(groups=groups, seed=12345)

        word_lists2 = {key: np.asarray(stim[key])[ids].tolist() for key in stim.keys()}

        # make sure control tokens do not appear in the target lists
        for k in stim.keys():
            assert ~np.any([set(t1).issubset(set(t2))
                            for t1, t2 in zip(word_lists1[k], word_lists2[k])])

    # this is repeat condition where the two lists are the same
    else:

        word_lists2 = word_lists1


    # ===== INITIATE EXPERIMENT CLASS ===== #

    # declare device and paths
    device = torch.device(argins.device if torch.cuda.is_available() else "cpu")

    # setup the model
    logging.info("Loading tokenizer {}".format(argins.path_to_tokenizer))
    tokenizer = AutoTokenizer.from_pretrained(argins.path_to_tokenizer)

    # ===== PREPARE INPUT SEQUENCES ===== #

    # this routing loops over prompts and prefixes
    # it keeps track of that in meta_data
    logging.info("Tokenizing and concatenating sequences...")
    input_sequences, input_sequences_unmasked, meta_data = concat_and_tokenize_inputs(prompt=prompts[argins.scenario]["1"],
                                                                                      prefix=prefixes[argins.scenario]["1"],
                                                                                      word_list1=word_lists1["n5"],
                                                                                      word_list2=word_lists2["n5"],
                                                                                      ngram_size="n5".strip("n"),
                                                                                      tokenizer=tokenizer,
                                                                                      ismlm=True)

    # save files
    savename = os.path.join(argins.output_dir, argins.output_filename + ".json")
    logging.info(f"Saving {savename}")
    with open(savename, "w") as fh:
        json.dump([e.tolist() for e in input_sequences], fh, indent=4)

    # if we have these, save them
    if input_sequences_unmasked:
        savename = os.path.join(argins.output_dir, argins.output_filename + "_unmasked" + ".json")
        logging.info(f"Saving {savename}")
        with open(savename, "w") as fh:
            json.dump([e.tolist() for e in input_sequences_unmasked], fh, indent=4)

    savename = os.path.join(argins.output_dir, argins.output_filename + "_info.json")
    logging.info(f"Saving {savename}")
    with open(savename, "w") as fh:
        json.dump(meta_data, fh, indent=4)

if __name__ == "__main__":

    main()