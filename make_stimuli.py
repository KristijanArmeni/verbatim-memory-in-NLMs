"""
make_stimuli.py contains utility functions to generate the dataset

Example usage:

ipython %run make_stimuli.py --which "toronto" \
                             --output_filename "test_filename.json"

"""

import pandas as pd
import random
import numpy as np
import sys
import json
import argparse

# uncomment this to debug
# which = ""
# permute = ""
# outputname = ""
parser = argparse.ArgumentParser()
parser.add_argument("--which", dtype=str, help="string, one of 'toronto' or 'semantic' or 'digit', specifies which stimulus set to build")
parser.add_argument("--output_filename", dtype=str)

argins = parser.parse_args()


# define w convenience function
def sample_words(word_list: list, sample_size: int) -> list:
    """
    sample_words(word_list, sample_size) splits <word_list> in to subsets
    of length <sample_size>
    """
    samples = []
    for x in range(0, len(word_list), sample_size):
        # append chunked list
        samples.append(word_list[x:(x + sample_size)])
    return samples


word_lists = None

# ===== PILOT DATA ===== #
if argins.which == "pilot":

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
elif argins.which == "toronto":

    # load in the toronto pool and rename columns to avoid blank spaces
    dat = pd.read_csv("./data/toronto_freq.txt", sep="\t", header=0). \
          rename(columns={"k-f freq": "k_f_freq"})

    lists = [dat.word.str.lower().to_numpy(),
             dat.concreteness.to_numpy(),
             dat.k_f_freq.to_numpy()]

    # utility function to keep same seed over several calls
    shuffle_ids = np.random.RandomState(123). \
                     permutation(np.arange(0, lists[0].size))

    word_lists = {"list-3": [sample_words(lists[i][shuffle_ids][0:60], 3) for i in range(len(lists))],
                  "list-5": [sample_words(lists[i][shuffle_ids][60:160], 5) for i in range(len(lists))],
                  "list-10": [sample_words(lists[i][shuffle_ids][160:360], 10) for i in range(len(lists))],
                  }

elif argins.which == "semantic_lists":

    df = pd.read_csv("./data/nouns_categorized.txt", header=None)
    print("Reading {} ...".format("./data/nouns_categorized.txt"))
    a = sample_words(df.loc[:, 0].str.lower().tolist(), 32)  # create a list of len(list[0])=32 tokens

    # generate lists of length 3, 5 and 10
    word_lists = {"list-3": [l[0:3] for l in a[0:20]],
                  "list-5": [l[3:8] for l in a[0:20]],
                  "list-10": [l[8:18] for l in a[0:20]]}


elif argins.which == "digit":

    # generate random numbers of certain lengths
    lens = [3, 5, 10]

    np.random.seed(54321)

    outlist = [np.random.random_integers(0, 10, l) for l in lens]

out = []
outlist = None
for list_size in word_lists.keys():

    # just a temporary patch, because for the non-semantic lists,
    # I have a tuple of words and frequency ratings
    # which is not true for the semantic lists
    if argins.which == "semantic_lists":

        outlist = word_lists[list_size]

    elif argins.which == "toronto":
        # here make sure to select strings, then convert numpy array to lists
        outlist = [lst.tolist() for lst in word_lists[list_size][0]]

    for k, l in enumerate(outlist):
        out.append(l)

with open(argins.output_filename, "w") as f:
    json.dump(out, f)