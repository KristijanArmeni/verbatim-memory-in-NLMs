"""
make_wordlists.py contains utility functions to generate the dataset

Example usage:

ipython %run make_wordlists.py --which "toronto" \
                               --output_filename "test_filename.json"

"""

import os
import pandas as pd
import numpy as np
import json
import argparse

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--which", type=str, choices=["random", "categorized", "ngrams-random"],
                    help="specifies which stimulus set to build")
parser.add_argument("--output_filename", type=str)

argins = parser.parse_args()

home_dir = os.environ["homepath"]

# define w convenience function
def sample_tokens(token_list: list, sample_size: int) -> list:
    """
    sample_words(word_list, sample_size) splits <word_list> in to subsets
    of length <sample_size>
    """
    samples = []

    for x in range(0, len(token_list), sample_size):
        # append chunked list
        samples.append(token_list[x:(x + sample_size)])
        
    return samples

def load_and_sample_noun_pool(path, which, model_vocab, n_items=None, n_lists=20, seed=None):
    """
    loads in the downloaded noun pools, filters all nouns that are not 
    in RNN vocab and returns an array
    """
    if which == "random":

        full_path = os.path.join(path, "toronto_freq.txt")
        
        # load in the toronto pool and rename columns to avoid blank spaces
        print("Reading {} ...".format(full_path))
        df = pd.read_csv(full_path, sep="\t", header=0).rename(columns={"k-f freq": "k_f_freq"})
    
        # make sure
        df['in_vocab'] = df['word'].str.lower().isin(model_vocab)  # determine whether a token is in vocabulary
        to_drop = df.loc[~df.in_vocab].index
        df.drop(to_drop, inplace=True)
        
        # convert data frame column to an array
        noun_pool = df.word.str.lower().to_numpy()

        # set seed
        rng = np.random.RandomState(seed)
        
        # construct random indices
        shuffle_ids = rng.permutation(np.arange(0, noun_pool.size))
    
        # split into chunks of n_lists
        
        # generate n_list lists of twenty random nouns
        token_sets = sample_tokens(token_list=noun_pool[shuffle_ids].tolist()[0:(20*n_lists)], 
                                   sample_size=20)
        
        # now return subset lists of desired size
        token_lists = [token_list[0:n_items] for token_list in token_sets]

    elif which == "categorized": 
        
        full_path = os.path.join(path, "nouns_categorized.txt")
        
        df = pd.read_csv(full_path, names=['token'])
        print("Reading {} ...".format(full_path))
    
        # check if they occur in rnn vocab
        df.loc[:, 'token'] = df.loc[:, 'token'].str.lower()
        df["token_id"] = np.tile(np.arange(0, 32), 32)   # construct indices for each token
        df['set_id'] = np.repeat(np.arange(0, 32), 32)   # construct set indices for each set
        df['in_vocab'] = df['token'].isin(model_vocab)         # determine whether a token is in vocabulary
    
        # count number of misses per list
        df['n_valid'] = df.groupby(['set_id']).in_vocab.transform('sum')
        df['valid_set'] = df.n_valid > 15           # make sure there are at least 17 in vocabulary tokens
        df['token_oov'] = df.token                  # add column with oov strings
        df.loc[~df.in_vocab, 'token_oov'] = 'oov'   # add oov tokens
    
        # and now grab the qualifying lists, filter out the oovs, then you have all lists with at least 18
        # tokens and sample from those lists
    
        to_drop = df.loc[~df.in_vocab].index
        
        # filtering function used downstream in filter()
        def notoov(element):
            return element != 'oov'
        
        # loop over sets that have at least 15 valid tokens
        # drop oov's and then grab 20 tokens from the list
        token_sets = []
        for v in df.loc[df.valid_set, 'set_id'].unique():
            token_sets.append(list(filter(notoov, df.loc[df.set_id == v, 'token_oov'].to_list()))[0:20])
            
        # generate lists of length 3, 5 and 10
        token_lists = [token_list[0:n_items] for token_list in token_sets[0:n_lists]]
        
    return token_lists


# ===== RUN CODE ===== #
vocab_file=os.path.join(home_dir, 'project', 'lm-mem', 'src', 'neural-complexity-master', "vocab.txt")

# read rnn vocab
with open(vocab_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    f.close()

rnn_vocab = [line.strip("\n") for line in lines]

# datadir with noun pool .txt files
path = os.path.join(home_dir, "project", "lm-mem", "src", "data")

# call load_and_sample_noun_pool repeatedly for each list size and
# store output list in a dict
which = argins.which
if argins.which=="ngrams-random":
    which="random"

out_dict = {"{}-gram".format(n_items): load_and_sample_noun_pool(path=path, 
                                                                 n_items=n_items, 
                                                                 n_lists=20,
                                                                 which=which, 
                                                                 model_vocab=rnn_vocab, 
                                                                 seed=12345)
            for n_items in [3, 5, 10]
            }

# if ngrams, use a different stimulus set
if argins.which == "ngrams-random":

    n_grams = [2, 3, 5, 7, 10]
    n_reps = 5
    
    # sample ngram sequences from 
    tmp = {"{}-gram".format(n_gram): [np.tile(alist[0:n_gram], reps=n_reps).tolist() 
                                      for alist in out_dict["10-gram"]]
                                      for n_gram in n_grams}
    
    out_dict = tmp


# ===== SAVE OUTPUT ===== #
out = []
outlist = None

out = [l for key in out_dict.keys() for l in out_dict[key]]

# now save the lists to .json files
with open(argins.output_filename, "w") as f:
    print("Writing {}".format(argins.output_filename))
    json.dump(out, f)
