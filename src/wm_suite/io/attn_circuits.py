import json
import os
import numpy as np
from typing import Tuple


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
        step = np.where(np.diff(vals) >= 1)[0][-1] + 1
        head_labels = scores["best_labels"][step]

    return head_labels
