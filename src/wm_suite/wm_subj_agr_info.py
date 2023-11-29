import os
import json
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Dict, Tuple
import torch
import logging
from tqdm import tqdm
from itertools import product
import inflect
from collections import Counter


from wm_subj_agr import read_json_data

logging.basicConfig(level=logging.INFO, format="%(message)s")

LINZEN2016 = "/scratch/ka2773/project/lm-mem/sv_agr/linzen2016/linzen2016_english.json"
LAKRETZ2021 = (
    "/scratch/ka2773/project/lm-mem/sv_agr/lakretz2021/short_nested_inner_english.json"
)

WIEGHTS_DIR = "/scratch/ka2773/project/lm-mem/output/ablation/"
ABLATION_KEYS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "01",
    "23",
    "0123",
    "89",
    "1011",
    "891011",
    "all",
]

MODEL_WEIGTHS = {
    f"ablate-{infix}": os.path.join(WIEGHTS_DIR, f"gpt2_ablate-{infix}-all.pt")
    for infix in ABLATION_KEYS
}
MODEL_WEIGTHS["unablated"] = "gpt2"
MODEL_WEIGTHS["rand-init"] = "rand-init"

d1 = read_json_data(LINZEN2016)

p = inflect.engine()

# Extract the first word in the target sequence
# one example has a type <carry'kayan'> for this reason we run an .split("'") on each token
strings = [
    [
        s.split()[0].split("'")[0].strip(",").strip(".")
        for s in list(e["target_scores"].keys())
    ]
    for e in d1
]
iscorr = [[i for i in list(e["target_scores"].values())] for e in d1]

# get the grammatical number of the first target (coded as 's' or 'p')
targets = [
    (str(p.compare_verbs(e[0], e[1])).split(":")[0], iscorr[i][0])
    for i, e in enumerate(strings)
    if len(e) == 2
]

# just check that we are indeed counting the correct targets (should be 499 of them)
assert sum([e[1] for e in targets]) == 499

# now count the number of target plural verbs by counting the number of 's' strings coming out of p.compare_verbs()
counts = Counter([e[0] for e in targets])


# get sentence statistics
sen_lens = [len(e["input"].strip().split(" ")) for e in d1]

d = {
    "mean": np.mean(sen_lens),
    "median": np.median(sen_lens),
    "std": np.std(sen_lens),
    "min": np.min(sen_lens),
    "max": np.max(sen_lens),
    "n_pl": counts["s"],
    "n_sg": counts["p"],
}

df = pd.DataFrame([d])
savepath = "/scratch/ka2773/project/lm-mem/sv_agr/linzen2016/descriptive.csv"
logging.info(f"Saving {savepath}")
df.to_csv(savepath, sep="\t")
