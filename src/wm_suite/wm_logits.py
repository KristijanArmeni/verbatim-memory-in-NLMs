
import json
import sys, os
from ast import literal_eval
from itertools import product
import numpy as np
import numpy.typing as npt
import pandas as pd
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# wm_suite (assumed you set PROJ_ROOT var and added it to path via set_path.sh)
from wm_suite.wm_ablation import ablate_attn_module
from wm_suite.wm_attention import find_cue_token_ids
from wm_suite.io.prepare_transformer_inputs import mark_subtoken_splits, make_word_lists, concat_and_tokenize_inputs
from wm_suite.io.stimuli import prefixes, prompts
from wm_suite.wm_test_suite import merge_states

import torch
from tqdm import trange
from typing import List, Dict
from tqdm import tqdm, trange
import logging


def get_logit_rank(inputs, model, at_timestep=-2) -> Dict:

    n_inputs = len(inputs)

    # structure for storing raw data
    out = {"logit": [],
           "rank": [],}

    # loop over inputs
    for i in trange(n_inputs, desc="sequence"):

        inp = inputs[i]
        print(inp.shape)
        output= model(inp, output_hidden_states=True)

        target_idx = inp[at_timestep + 1]                     # index of the upcoming token
        logit = output.logits[at_timestep, target_idx].cpu().item()

        # compute rank too
        probs = torch.softmax(output.logits[at_timestep, :], dim=0)

        # sort such that the first element is the token with highest probability
        _, idxs1 = torch.sort(probs, descending=True)

        rank = torch.where(idxs1 == target_idx)[0].cpu().item()

        out["logit"].append(logit)
        out["rank"].append(int(rank))

    return out


def get_test_input_args():

    inps = ["--layer_head_dict", str({0: [0, 1, 2], 5: [5, 8, 9]}),
            "--noun_list_file", "/home/ka2773/project/lm-mem/src/data/noun_lists/random_lists.json"]

    return inps


def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gpt2")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--layer_head_dict", type=str,
                        help="A string specifying a python dict indicate what layers/heads to ablate (e.g"
                        "'{0: [1, 2, 3]}')")
    parser.add_argument("--list_len", type=int, default=3)
    parser.add_argument("--prompt_len", type=str, default="1")
    parser.add_argument("--noun_list_file", type=str)
    parser.add_argument("--output_name")
    parser.add_argument("--output_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #input_args = get_test_input_args()

    if input_args:
        argins = parser.parse_args(input_args)
    else:
        argins = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(argins.tokenizer)
    model = GPT2LMHeadModel.from_pretrained(argins.checkpoint)


    if argins.layer_head_dict:

        layer_head_dict = literal_eval(argins.layer_head_dict)
        model = ablate_attn_module(model, layer_head_dict=layer_head_dict, ablation_type="zero")
 

    # ===== PREPARE INPUTS ===== #

    # fname = os.path.join(data_dir, argins.input_filename)
    word_lists1, word_lists2 = make_word_lists(argins.noun_list_file, condition="repeat")

    # this routine loops over prompts and prefixes
    # it keeps track of that in meta_data
    logging.info("Tokenizing and concatenating sequences...")
    scenario = "sce1"
    prompt_len = argins.prompt_len
    list_len = argins.list_len
    input_sequences, input_sequences_info = concat_and_tokenize_inputs(prompt=prompts[scenario][prompt_len],
                                                                       prefix=prefixes[scenario]["1"],
                                                                       word_list1=word_lists1[f"n{list_len}"],
                                                                       word_list2=word_lists2[f"n{list_len}"],
                                                                       ngram_size=str(list_len),
                                                                       pretokenize_moses=False,
                                                                       tokenizer=tokenizer,
                                                                       bpe_split_marker="Ġ",
                                                                       marker_logic="outside",
                                                                       ismlm=False)

    # add prompt field to the dict
    input_sequences_info["prompt"] = [argins.prompt_len for _ in input_sequences_info['list_len']]


    # find index of the query token (':') and add one, to make the first noutn end of the sequence
    query_idxs = [find_cue_token_ids(np.array(markers))[1] + 1 for markers in input_sequences_info["trialID"]]


    model.to(device)

    # sample inputs (sample +1 from query to have sequence include the first noun)
    inputs = [inp[0, 0:(query_idxs[i]+1)].to(device) for i, inp in enumerate(input_sequences)]

    # get ranks and logits from the pre-final timestep (i.e. just before the first noun, which is the last item in the sequence)
    output = get_logit_rank(inputs, model, at_timestep=-2)

    if argins.layer_head_dict:
        output["lh_dict"] = str(layer_head_dict)
    else:
        output["lh_dict"] = "unablated"


    if argins.output_dir:
        fn = os.path.join(argins.output_dir, argins.output_name)
        logging.info(f"Saving {fn}")
        with open(fn, "w") as fh:
            json.dump(output, fh)


if __name__ == "__main__":

    main()

