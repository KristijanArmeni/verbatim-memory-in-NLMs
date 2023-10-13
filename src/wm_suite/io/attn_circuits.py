import json
import os
import numpy as np
from typing import Tuple

from wm_suite.io.prepare_transformer_inputs import get_input_sequences


first_sequence_nouns = {
        "n1": 'Ġpatience',
        "n2": 'Ġnotion',
        "n3": 'Ġmovie',
        "n4": 'Ġwomen',
        "n5": "Ġcanoe",
        "n6": 'Ġnovel',
        "n7": 'Ġfolly',
        "n8": 'Ġsilver',
        "n9": 'Ġeagle',
        "n10": 'Ġcenter',
    }


def get_query_target_indices(list_len:str, which:str) -> Tuple:
    """
    A helper function to get the indices of the query and target words in the input sequence.

    """

    _, _, seqs = get_input_sequences(condition="repeat", scenario="sce1", list_type="random", list_len=list_len, prompt_key="1", 
                                     tokenizer_name="gpt2", pretokenize_moses=False)

    # define indices based on tokens in the first sequences (has no BPE-split tokens)
    t = np.array(seqs[0].toks[0])

    nouns = list(first_sequence_nouns.values())[0:int(list_len[-1])]
    codes = list(first_sequence_nouns.keys())[0:int(list_len[-1])]

    if which == "match":
        query_ids = {c: (n, np.where(t == n)[0][-1]) for n, c in zip(nouns, codes)}
        target_ids = {c: (n, np.where(t == n)[0][0]) for n, c in zip(nouns, codes)}

    elif which == "postmatch":
        increment = 1
        query_ids = {c: (n, np.where(t == n)[0][-1]) for n, c in zip(nouns, codes)}
        target_ids = {c: (t[(np.where(t == n)[0][0] + increment)], (np.where(t == n)[0][0]) + increment) for n, c in zip(nouns, codes)}

    elif which == "recent":
        increment = -1
        query_ids = {c: (n, np.where(t == n)[0][-1]) for n, c in zip(nouns, codes)}
        target_ids = {c: (t[np.where(t == n)[0][0] + increment], (np.where(t == n)[0][-1] + increment)) for n, c in zip(nouns, codes)}

    return {"queries": query_ids, "targets": target_ids}


def get_circuit(which: str):

    if which == "random_144":
        # get the attention weights for that sequence
        d = "/scratch/ka2773/project/lm-mem/output/ablation/zero_attn/topk/search"
        fn = "scores_all_heads_s2s3_iter-20.json"
        with open(os.path.join(d, fn), "r") as f:
            scores = json.load(f)

        # we now take the list of heads that gave us 100% repeat surprisal
        vals = np.array(scores["rs"]["scores"])
        step = np.where(vals > 100)[0][0]
        head_labels = scores["best_labels"][step]

    elif which == "targeted_ablation":

        # get the attention weights for that sequence
        d = "/scratch/ka2773/project/lm-mem/output/ablation/zero_attn/topk/search"
        fn = "greedy_search_joint_topk_n2_n3_orig.json"
        with open(os.path.join(d, fn), "r") as f:
            scores = json.load(f)

        # we now take the list of heads that gave us 100% repeat surprisal
        vals = np.array(scores["rs"]["scores"])
        step = np.where(np.diff(vals) >= 1)[0][-1]+1
        head_labels = scores["best_labels"][step]

    return head_labels
