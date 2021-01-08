"""
make_stimuli.py contains utility functions to generate the dataset
"""

import pandas as pd
import random
import numpy as np
import sys
import json

# uncomment this to debug
# which = ""
# permute = ""
# outputname = ""

which = sys.argv[1]
permute = bool(int(sys.argv[2]))
outputname = sys.argv[2]

# ===== PILOT DATA ===== #
if which == "pilot":

    df = pd.read_csv("subtlex_us_pos.txt", sep="\t")

    wfmax = df.SUBTLWF.quantile(1)
    wfmin = df.SUBTLWF.quantile(0.975)

    sel = (df.Dom_PoS_SUBTLEX == "Noun") & (df.SUBTLWF >= wfmin)
    nouns = df.loc[sel, "Word"]

    random.seed(123, 2)
    wlist = np.asarray(random.sample(nouns.tolist(), 30))

    # generate set indices
    sets = np.repeat(np.arange(1, 4), 10)

    # save it
    out = pd.DataFrame(np.vstack((sets, wlist)).T, columns=["set", "token"])
    out.to_csv("./data/stimuli_random.txt", index=False)

    # ===== MAKE WORD PREDICTION STIMULI ===== #
if which == "toronto":

    # load in the toronto pool and rename columns to avoid blank spaces
    dat = pd.read_csv("./data/toronto_freq.txt", sep="\t", header=0). \
          rename(columns={"k-f freq": "k_f_freq"})

    lists = [dat.word.str.lower().to_numpy(),
             dat.concreteness.to_numpy(),
             dat.k_f_freq.to_numpy()]

    lengths = ()

    # utility function to keep same seed over several calls
    shuffle_ids = np.random.RandomState(123). \
                     permutation(np.arange(0, lists[0].size))

    def sample_words(word_list: list, sample_size: int) -> list:
        """
        sample_words(word_list, sample_size, random_seed) splits <word_list> in to subsets
        of length <sample_size>
        """
        samples = []
        for x in range(0, len(word_list), sample_size):
            # append chunked list
            samples.append(word_list[x:(x+sample_size)])
        return samples

    word_lists = {"list-3": [sample_words(lists[i][shuffle_ids][0:60], 3) for i in range(len(lists))],
                  "list-5": [sample_words(lists[i][shuffle_ids][60:160], 5) for i in range(len(lists))],
                  "list-10": [sample_words(lists[i][shuffle_ids][160:360], 10) for i in range(len(lists))],
                  }

    out = []
    for list_size in word_lists.keys():
        for k, l in enumerate(word_lists[list_size][0]):
            out.append(l.tolist())


    with open(outputname, "w") as f:
        json.dump(out, f)