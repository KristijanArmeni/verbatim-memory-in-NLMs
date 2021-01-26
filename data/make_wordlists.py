"""
make_wordlists.py contains utility functions to generate the dataset

Example usage:

ipython %run make_wordlists.py --which "toronto" \
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
parser.add_argument("--which", dtype=str,
                    help="string, one of 'random' or 'categorized', specifies which stimulus set to build")
parser.add_argument("--output_filename", dtype=str)

argins = parser.parse_args()


# define w convenience function
def sample_words(word_list: list, sample_size: int, ignore='OOV') -> list:
    """
    sample_words(word_list, sample_size) splits <word_list> in to subsets
    of length <sample_size>
    """
    samples = []

    for x in range(0, len(word_list), sample_size):
        # append chunked list
        samples.append(word_list[x:(x + sample_size)])
    return samples

# read rnn vocab
with open('./neural-complexity-master/vocab.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    f.close()

vocab = [line.strip("\n") for line in lines]

word_lists = None


# filtering function used downstream in filter()
def notoov(element):
    return element != 'oov'

# ===== PILOT DATA ===== #
if argins.which == "pilot":

    df = pd.read_csv("subtlex_us_pos.txt", sep="\t")

    # we sample words at the top of the frequency distribution
    wfmax = df.SUBTLWF.quantile(1)
    wfmin = df.SUBTLWF.quantile(0.975)

    # select nouns and apply the lower bound criterion
    sel = (df.Dom_PoS_SUBTLEX == "Noun") & (df.SUBTLWF >= wfmin)
    nouns = df.loc[sel, "Word"]

    # sample randomly from the list of nouns
    random.seed(123, 2)
    wlist = np.asarray(random.sample(nouns.tolist(), 30))

    # generate set indices
    sets = np.repeat(np.arange(1, 4), 10)

    # save it
    out = pd.DataFrame(np.vstack((sets, wlist)).T, columns=["set", "token"])
    out.to_csv("./data/stimuli_random.txt", index=False)

    # ===== MAKE WORD PREDICTION STIMULI ===== #
elif argins.which == "random":

    # load in the toronto pool and rename columns to avoid blank spaces
    df = pd.read_csv("./data/toronto_freq.txt", sep="\t", header=0). \
          rename(columns={"k-f freq": "k_f_freq"})

    # filter possible oovs
    df['in_vocab'] = df['word'].str.lower().isin(vocab)         # determine whether a token is in vocabulary
    to_drop = df.loc[~df.in_vocab].index
    df.drop(to_drop, inplace=True)

    lists = [df.word.str.lower().to_numpy(),
             df.concreteness.to_numpy(),
             df.k_f_freq.to_numpy()]

    # utility function to keep same seed over several calls
    shuffle_ids = np.random.RandomState(123). \
                     permutation(np.arange(0, lists[0].size))

    word_lists = {"list-3": [sample_words(lists[i][shuffle_ids][0:60], 3) for i in range(len(lists))],
                  "list-5": [sample_words(lists[i][shuffle_ids][60:160], 5) for i in range(len(lists))],
                  "list-10": [sample_words(lists[i][shuffle_ids][160:360], 10) for i in range(len(lists))],
                  }

elif argins.which == "categorized":

    df = pd.read_csv("./data/nouns_categorized.txt", names=['token'])
    print("Reading {} ...".format("./data/nouns_categorized.txt"))

    # check if they occur in rnn vocab
    df.loc[:, 'token'] = df.loc[:, 'token'].str.lower()
    df["token_id"] = np.tile(np.arange(0, 32), 32)   # construct indices for each token
    df['set_id'] = np.repeat(np.arange(0, 32), 32)   # construct set indices for each set
    df['in_vocab'] = df['token'].isin(vocab)         # determine whether a token is in vocabulary

    # count number of misses per list
    df['n_valid'] = df.groupby(['set_id']).in_vocab.transform('sum')
    df['valid_set'] = df.n_valid > 17           # make sure there are at least 17 in vocabulary tokens
    df['token_oov'] = df.token                  # add column with oov strings
    df.loc[~df.in_vocab, 'token_oov'] = 'oov'   # add oov tokens

    # and now grab the qualifying lists, filter out the oovs, then you have all lists with at least 18
    # tokens and sample from those lists

    to_drop = df.loc[~df.in_vocab].index

    # loop over sets that have at least 18 valid tokens
    # drop oov's and then grab 20 tokens from the list
    toks = []
    for v in df.loc[df.valid_set, 'set_id'].unique():
        toks.append(list(filter(notoov, df.loc[df.set_id == v, 'token'].to_list()))[0:20])

    # generate lists of length 3, 5 and 10
    word_lists = {"list-3": [l[0:3] for l in toks[0:20]],
                  "list-5": [l[3:8] for l in toks[0:20]],
                  "list-10": [l[8:18] for l in toks[0:20]]}

out = []
outlist = None
for list_size in word_lists.keys():

    # just a temporary patch, because for the non-semantic lists,
    # I have a tuple of words and frequency ratings
    # which is not true for the semantic lists
    if argins.which == "categorized":

        outlist = word_lists[list_size]

    elif argins.which == "random":
        # here make sure to select strings, then convert numpy array to lists
        outlist = [lst.tolist() for lst in word_lists[list_size][0]]

    for k, l in enumerate(outlist):
        out.append(l)

# now save the lists to .json files
with open(argins.output_filename, "w") as f:
    json.dump(out, f)