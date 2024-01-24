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
import warnings
from ..paths import DATA_PATH, ROOT_PATH


# define w convenience function
def sample_tokens(token_list: list, step_size: int, window_size: int) -> list:
    """
    sample_words(word_list, sample_size) splits <word_list> in to subsets
    of length <sample_size>
    """
    samples = []
    
    if window_size > step_size:
        warnings.warn("Window size larger than step size. Output samples "
                      "will have overlapping samples")
    
    for x in range(0, len(token_list), step_size):
        # append chunked list
        samples.append(token_list[x:(x + window_size)])
        
    return samples


def chunk(lst, n, step):
    """
    same as sample_tokens, but yields chunks as lists.
    """
    for i in range(0, len(lst), step):
        yield lst[i:i + n]


def load_and_sample_noun_pool(path, which, model_vocab, n_items=None, n_lists=20, seed=None):
    """
    loads in the downloaded noun pools, filters all nouns that are not 
    in RNN vocab and returns an array
    """
    
    if which == "random":

        full_path = os.path.join(path, "nouns_arbitrary.txt")
        
        # load in the toronto pool and rename columns to avoid blank spaces
        print("Reading {} ...".format(full_path))
        df = pd.read_csv(full_path, sep="\t", header=0).rename(columns={"k-f freq": "k_f_freq"})
    
        # make sure that selected pools are in the RNN vocabulary
        df['in_vocab'] = df['word'].str.lower().isin(model_vocab)  
        to_drop = df.loc[~df.in_vocab].index
        df.drop(to_drop, inplace=True)
        
        # convert data frame column to an array
        noun_pool = df.word.str.lower().to_numpy()

        # set seed
        rng = np.random.RandomState(seed)
        
        # construct random indices
        shuffle_ids = rng.permutation(np.arange(0, noun_pool.size))
    
        # split into chunks of n_lists
        
        # generate n_list lists of n_items random nouns
        token_lists = sample_tokens(token_list=noun_pool[shuffle_ids].tolist()[0:(n_items*n_lists)], 
                                    step_size=n_items,
                                    window_size=n_items)
        

    elif which == "categorized": 
        
        full_path = os.path.join(path, "nouns_categorized.txt")
        
        df = pd.read_csv(full_path, names=['token'])
        print("Reading {} ...".format(full_path))
        
        # there are 32 sets of 32 tokens in total
        n_sets = 32
        n_toks_per_set = 32
        
        # check if they occur in rnn vocab
        df.loc[:, 'token'] = df.loc[:, 'token'].str.lower()
        df["token_id"] = np.tile(np.arange(0, n_toks_per_set), n_sets)   # construct indices for each token
        df['set_id'] = np.repeat(np.arange(0, n_sets), n_toks_per_set)   # construct set indices for each set
        df['in_vocab'] = df['token'].isin(model_vocab)           # determine whether a token is in vocabulary
        
        # only those sets qualify that have at least threshold
        # number of in vocabulary tokens (that's how long our lists will be)
        threshold = n_items
        
        # count number of misses per list
        df['n_valid'] = df.groupby(['set_id']).in_vocab.transform('sum')
        df['valid_set'] = df.n_valid > threshold           # make sure there are at least 17 in vocabulary tokens
        df['token_oov'] = df.token                  # add column with oov strings
        df.loc[~df.in_vocab, 'token_oov'] = 'oov'   # mark the oov tokens
        
        # filtering function used downstream in filter()
        def notoov(element):
            return element != 'oov'
        
        # get the ids for the valid sets, print some feedback
        valid_sets = df.loc[df.valid_set, 'set_id'].unique()
        print("There are {} semantic token sets with at least {} in-vocabulary tokens".format(len(valid_sets), threshold))
        print("Returning {} lists of length {}.".format(len(valid_sets), n_items))
        
        # loop over sets that have at least n_items valid tokens
        # drop oov's and then grab n_tiems tokens from the list
        token_lists = []
        for v in valid_sets[0:n_lists]:
            
            toks = df.loc[df.set_id == v, 'token_oov'].to_list()
            filtered_toks = list(filter(notoov, toks))
            
            token_lists.append(filtered_toks[0:n_items])
        
    return token_lists


def get_noun_lists_toronto(which:str, list_length:int=10):

    vocab_file=os.path.join(ROOT_PATH, 'src', 'rnn', "vocab.txt")

    # read rnn vocab
    with open(vocab_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()

    rnn_vocab = [line.strip("\n") for line in lines]

    # datadir with noun pool .txt files
    path = os.path.join(ROOT_PATH, "src", "data")
        
    # construct distractors from random noun pools
    if which == "ngram-distractors" or which == "ngram-random":
        which = "random"
    elif which == "ngram-categorized":
        which = "categorized"

    # ===== CREATE WORD LISTS AND N-GRAM SUBSETS ===== #
    lists_of_tokens = load_and_sample_noun_pool(path=path, 
                                                n_items=list_length, 
                                                n_lists=23,
                                                which=which, 
                                                model_vocab=rnn_vocab, 
                                                seed=12345)

    # create circular shifts of the lists
    cshifts = list(np.arange(0, list_length))

    lists_of_tokens_shifted = [np.roll(np.asarray(alist), s).tolist() for alist in lists_of_tokens
                            for s in cshifts]

    # generate lists of length 3, 5, 7 and 10
    # now create subsets of the original list
    subsets = [3, 5, 7, 10]

    # first sample angain the random noun pool
    out_dict = {"n{}".format(n_items): [alist[0:n_items] for alist in lists_of_tokens_shifted]
                for n_items in subsets}

    n_grams = [2, 3, 5, 7, 10]
    n_reps = 5

    # for ngram lists sample the created lists repeatedly
    if which in ["ngram-random", "ngram-categorized"] or which=="ngram-distractors":
            
        # sample repeated ngram sequences from the lists of 10 items
        tmp = {"n{}".format(n_gram): [list(chunk(np.tile(alist[0:n_gram], reps=n_reps).tolist(), n_gram, n_gram)) 
                                        for alist in out_dict["n10"]]
                                        for n_gram in n_grams}

    # for ngram lists sample the created lists repeatedly
    elif which == "ngram-control":
        
        
        
        # sample repeated ngram sequences from the lists of 10 items
        tmp = {"n{}".format(n_gram): [list(chunk(np.tile(alist[0:n_gram], reps=n_reps).tolist(), n_gram, n_gram)) 
                                        for alist in out_dict["n10"]]
                                        for n_gram in n_grams}

        
        out_dict = tmp

    elif which == "ngram-distractors":
        
        # create extra 4 lists which will be used for the
        # interleaved items
        lists_of_tokens = load_and_sample_noun_pool(path=path, 
                                                    n_items=10, 
                                                    n_lists=26,
                                                    which="random", 
                                                    model_vocab=rnn_vocab, 
                                                    seed=12345)
        
        # store the pool of distractor nouns
        # (not used for the regular lists)
        distractor_set = [el for lst in lists_of_tokens[23::] for el in lst]
        
        # there needs to be n-1 interleaved items
        ngram_reps = 5
        n_reps_distractors = ngram_reps - 1

        # sample repeated ngram sequences from the lists of 10 items
        out_dict = {"n{}".format(size): [list(chunk(thelist[0:(n_reps_distractors*7)], size, 7)) 
                                                for thelist in [distractor_set]]
                                                for size in [2, 3, 5, 7]}

    return out_dict


def load_noun_lists_concrete_abstract(which:str):

    if which == "concrete":
        filename = "nouns_500_conc.txt"
    elif which == "abstract":
        filename = "nouns_500_abst.txt"

    # ===== CREATE WORD LISTS AND N-GRAM SUBSETS ===== #
    fn = os.path.join(DATA_PATH, "abst-conc", "abst-conc-extremes", filename)
    df = pd.read_csv(fn, sep="\t", header=None)
    df.columns = ["word", "conc", "freq"]

    return df.word.tolist()


def chunk_lists_and_circular_shift(noun_list: list, list_length:int=10) -> dict:
    """
    Take a list of nouns and chunk them into lists of length <list_length> and
    then circularly shift each list such that all list items occur in all serial positoins.
    
    Parameters
    ----------
    noun_list : list
        list of nouns
    list_length : int
        length of the lists to be generated
    
    Returns
    -------
    dict
        dictionary with keys "n<list_len>", and values lists of lists
    
    """
    # generate n_list lists of `list_len` random nouns
    lists_of_nouns = sample_tokens(
        token_list=noun_list,
        step_size=list_length, 
        window_size=list_length,
    )

    # create circular shifts of the lists
    cshifts = list(np.arange(0, list_length))

    lists_of_tokens_shifted = [
        np.roll(np.asarray(alist), s).tolist() for alist in lists_of_nouns
        for s in cshifts
    ]

    # make sure all are the same length (we drop residual, shorter lists)
    lists_of_tokens_shifted = [e for e in lists_of_tokens_shifted if len(e) == list_length]

    # return a dict
    return {"n{}".format(list_length): lists_of_tokens_shifted}    


def main():
    """
    main function
    """
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, choices=["random", "categorized", 
                                                    "ngram-random", "ngram-categorized",
                                                    "ngram-distractors", "concrete", "abstract"],
                        help="specifies which stimulus set to build")
    parser.add_argument("--output_dir", type=str)

    argins = parser.parse_args()

    if argins.which in ["concrete", "abstract"]:

        nouns = load_noun_lists_concrete_abstract(which=argins.which)
        out_dict = chunk_lists_and_circular_shift(nouns, list_length=10)

            # now save the lists to .json files
        filename = f"nouns_{argins.which}.json"
        fn = os.path.join(argins.output_dir, filename)
        with open(fn, "w") as f:
            print(f"Writing {fn}")
            json.dump(out_dict, f)


    elif argins.which in ["random", "categorized", "ngram-random", "ngram-categorized", "ngram-distractors"]:

        out_dict = get_noun_lists_toronto(which=argins.which, list_length=3)

        # now save the lists to .json files
        fn = os.path.join(argins.output_dir, "nouns_{}.json".format(argins.which))
        with open(fn, "w") as f:
            print(f"Writing {fn}")
            json.dump(out_dict, f)


if __name__ == "__main__":

   main()


