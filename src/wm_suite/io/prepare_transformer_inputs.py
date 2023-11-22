import argparse
import json
import os
import re
import sys
from string import punctuation
from typing import Dict, List, Tuple

import numpy as np
import torch
from mosestokenizer import MosesTokenizer
from transformers import AutoTokenizer

from ..paths import DATA_PATH, get_paths
from ..utils import logger
from .stimuli import prefixes, prompts


# ===== WRAPPERS FOR DATASET CONSTRUCTION ===== #


def mark_subtoken_splits(
    tokens: List[str], split_marker: str, marker_logic: str, eos_markers: List[str]
) -> List:
    """
    mark_subtoken_splits() keeps track of whether or not a token was
    subplit into subwords or not.
    Each token is counted and if it was split, the subtokens are
    counted as a single token.
    E.g. the tokens ["I", "saw", "a", "sp", "Ġarrow", "yesterday"]
    would be coded as [0, 1, 2, 3, 3, 4] indicating that there are 4
    unique tokens and that token nr 3 (sparrow) was split into two
    subtokens (sp + arrow).

    Parameters:
    ----------
    tokens : list
        list of strings, containing the split tokens
    split_markers : string
        string denoting

    Returns :
    -------
    ids : list
        list containing indices that mark groups (prefix, list1,
        prompt, list2) withing each list

    Example:
    -------
    ids = mark_subtoken_splits(tokens=["I", "saw", "a", "sp", "Ġarrow"],
                               split_marker="Ġ",
                               marker_logic="outside",
                               eos_markers: ["<|endoftext|>", "<|endoftext|>"])

    """
    ids = []
    count = 0

    # some tokenizers will mark a split by omitting the split_marker,
    # in this case you have to stop counting when split_marker is
    # missing
    count_token = None
    if marker_logic == "outside":
        count_token = [split_marker in token for token in tokens]
    # in other cases (e.g. for BERT), you have to stop counting when
    # split_marker is present
    elif marker_logic == "within":
        count_token = [
            (split_marker not in token)
            and (token not in list(punctuation) + eos_markers)
            for token in tokens
        ]

    # for tokenizers that do not do BPE tokenization we count each
    # token and ignore punctuation plus eos markers
    if split_marker is None:
        count_token = [
            (True and (token not in list(punctuation) + eos_markers))
            for token in tokens
        ]

    # loop over tokens and apply the counting logic
    for i in range(len(tokens)):
        # if the token is split, it does not gave the symbol for whitespace
        # if the word is at position 0, it also has to start a new token, so
        # count it
        if count_token[i]:
            count += 1

        if tokens[i] in punctuation:
            ids.append(-1)  # code punctuation with -1
        elif tokens[i] in eos_markers:
            ids.append(-2)  # code eos strings with -2
        else:
            ids.append(count)

    return ids


class InputSequence(object):
    def __init__(self):
        self.str = None
        self.toks = None
        self.ids = None
        self.stim_id = None
        self.trial_ids = None
        self.subtok_ids = None
        self.position_ids = None
        self.list_len = None
        self.prompt = None

    def __repr__(self) -> str:
        return (
            f"InputSequence({self.stim_id:02d}, "
            f"list_len={self.list_len}, "
            f"n_toks={len(self.toks)})"
        )


def assemble_sequence(tokenizer, prefix, list1, prompt, list2):
    seq_parts = []
    seq_codes = []

    def clen():
        return len(seq_parts[-1])

    # prefix
    seq_parts.append(prefix + " ")
    seq_codes.extend([0] * clen())
    # items
    for item in list1[:-1]:
        seq_parts.append(item)
        seq_codes.extend([1] * clen())
        seq_parts.append(", ")
        seq_codes.extend([-1] * clen())
    seq_parts.append(list1[-1])
    seq_codes.extend([1] * clen())
    seq_parts.append(".")
    seq_codes.extend([-1] * clen())
    # prompt
    seq_parts.append(" " + prompt + " ")
    seq_codes.extend([2] * clen())
    # items2
    for item in list2[:-1]:
        seq_parts.append(item)
        seq_codes.extend([3] * clen())
        seq_parts.append(", ")
        seq_codes.extend([-1] * clen())
    seq_parts.append(list2[-1])
    seq_codes.extend([3] * clen())
    seq_parts.append(".")
    seq_codes.extend([-1] * clen())
    seq = "".join(seq_parts)
    tokenized = tokenizer(seq, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    input_ids = tokenized["input_ids"]

    char_codes = np.zeros(len(seq), dtype=np.int64) - 1
    for idx, match_ in enumerate(re.finditer(r"\w*-?\w+-?'?\w*'?", seq), 1):
        char_codes[match_.start() : match_.end()] = idx

    token_codes = []
    word_codes = []
    for a, b in offsets:
        # c = -1 if a == b else seq_codes[a]
        # print(a, b, seq[a:b], c, char_codes[a:b])
        if a == b:
            token_codes.append(-1)
            word_codes.append(-1)
        else:
            assert all(c == seq_codes[a] for c in seq_codes[a:b])
            token_codes.append(seq_codes[a])
            word_codes.append(char_codes[a])

    return seq, input_ids, token_codes, word_codes


def concat_and_tokenize_inputs(
    prefix: str,
    prompt: str,
    word_list1: List[str],
    word_list2: List[str],
    ngram_size: str,
    pretokenize_moses: bool = False,
    tokenizer=None,
) -> Tuple[List[torch.Tensor], Dict]:
    """concat_and_tokenize_inputs() concatenates and tokenizes
    strings as inputs for wm_suite.Experiment() class

    Parameters:
    ----------
    prefix : str
        a string that preceeds the first noun list
    prompt : str
        a string that follows the second noun list
    word_list1 : list of lists
        a list of lists, each element in the list is a list of nouns,
        these lists are first lists in the sequence
    word_list2 : list
        a list of lists, each element in the list is alist of nouns,
        these lists are second lists in the sequence
    ngram_size : str
        a string indicating number of elements in entries of
        word_list1 and word_list2
    bpe_split_marker : str
        a string that is used by HuggingFace tokenizer classes to mark
        tokens that were split into BPEs
    marker_logic : str ("outside", "within")
        string indicating whether bpe_split_marker indicates outer BPE
        tokens in split tokens (like GPT2) or inner (like BERT)
    ismlm : boolean
        inicates whether or not the current model is a masked language
        model or not

    Returns:
    -------
    input_seqs_tokenized : list of tensors
        input sequences to be fed to the models
    metadata : dict
        dict containing information about stimuli, it contains fields 'stimid', 'trialID', 'positionID', 'subtok', 'list_len'

    """
    input_seqs_new = []
    metadata = {
        "stimid": [],
        "trialID": [],
        "positionID": [],
        "subtok": [],
        "list_len": [],
    }

    if pretokenize_moses:
        logger.info("Pretokenizing input sequenes with MosesTokenizer('en') ...")
        with MosesTokenizer("en") as pretokenize:
            prefix = " ".join(pretokenize(prefix)) + " "
            prompt = " ".join(pretokenize(prompt)) + " "
            word_list1 = [
                [" ".join(pretokenize(w)) for w in words] for words in word_list1
            ]
            word_list2 = [
                [" ".join(pretokenize(w)) for w in words] for words in word_list2
            ]

    # loop over trials
    for words1, words2 in zip(word_list1, word_list2):
        inp_seq = InputSequence()
        seq, input_ids, token_codes, word_codes = assemble_sequence(
            tokenizer, prefix, words1, prompt, words2
        )

        inp_seq.str = seq
        inp_seq.ids = torch.tensor(input_ids, dtype=torch.int64)
        inp_seq.toks = tokenizer.convert_ids_to_tokens(input_ids)
        metadata["trialID"] = token_codes
        metadata["stimid"].append(len(input_seqs_new))
        metadata["subtok"] = word_codes
        metadata["list_len"].append(ngram_size)

        inp_seq.list_len = ngram_size
        inp_seq.stim_id = len(input_seqs_new)
        inp_seq.trial_ids = token_codes
        inp_seq.subtok_ids = word_codes

        input_seqs_new.append(inp_seq)

    return input_seqs_new


def sample_indices_by_group(groups: np.ndarray, seed: int) -> np.ndarray:
    """
    randomized_indices = sample_indices_by_group(groups, seed)

    Parameters:
    ----------
        groups : np.array,
            array defining group membership (e.g. [0, 0, 0, 1, 1, 1])
        seed :
            int, argument for np.random.RandomState

    Returns:
    -------
        randomized_indices : np.array
            randomly sampled indices of groups.size

    Helper function that creates randomized indices form
    np.arange(groups.size) by following the structure of group
    elements in group. It ensures that every element groups is paired
    with an element outside its own group.
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

        sel_ids = rng.choice(a=candidate_pool, size=sample_size)
        out_ids[groups == group] = sel_ids

        # mark already selected indices
        indices[sel_ids] = ignore_id

    return out_ids


def ensure_list2_notequal(
    list1: List, list2: List, start_seed: int, seed_increment: int
) -> List:
    """
    Parameters:
    ----------

    Returns:
    -------
    new_list2 = ensure_list2_notequal(list1, list2, start_seed, seed_increment)

    Helper function that ensures that all elements in list2 are not
    equal to elements in list1 by iteratively applying new permutations
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
            list2_new = [
                rng.permutation(l2).tolist() if l1 == l2 else l2
                for l1, l2 in zip(list1, list2)
            ]

            # update the criterion condition (none should be equal)
            are_equal = [[l1 == l2] for l1, l2 in zip(list1, list2_new)]

            # update seed for a new try
            seed += seed_increment

    return list2_new


def make_word_lists(inputs_file: str, condition: str) -> Tuple[List, List]:
    logger.info("Loading {} ...".format(inputs_file))
    with open(inputs_file) as f:
        stim = json.load(f)

    word_lists1 = stim

    if condition == "permute":
        # This condition test for the effect of word order
        # Lists have the same words, but the word order is permuted
        # int the second one
        word_lists2 = {
            key: [
                np.random.RandomState((543 + j) * 5).permutation(stim[key][j]).tolist()
                for j in range(len(stim[key]))
            ]
            for key in stim.keys()
        }

        for list_size in word_lists2.keys():
            word_lists2[list_size] = ensure_list2_notequal(
                list1=word_lists1[list_size],
                list2=word_lists2[list_size],
                start_seed=123,
                seed_increment=10,
            )

        # make sure control tokens do not appear in the target lists
        for k in stim.keys():
            assert ~np.any(
                [[t1 == t2] for t1, t2 in zip(word_lists1[k], word_lists2[k])]
            )

    elif condition == "control":
        # This serves as a control conditions
        # Here list length is the only common factor between two lists

        logger.info("Creating control condition...")

        n_items_per_group = 10
        n_groups = len(word_lists1["n10"]) // n_items_per_group
        groups = np.repeat(np.arange(0, n_groups), n_items_per_group)

        ids = sample_indices_by_group(groups=groups, seed=12345)

        word_lists2 = {key: np.asarray(stim[key])[ids].tolist() for key in stim.keys()}

        # make sure control tokens do not appear in the target lists
        for k in stim.keys():
            assert ~np.any(
                [
                    set(t1).issubset(set(t2))
                    for t1, t2 in zip(word_lists1[k], word_lists2[k])
                ]
            )

    # this is repeat condition where the two lists are the same
    else:
        word_lists2 = word_lists1

    return word_lists1, word_lists2


def get_input_sequences(
    condition: str = "repeat",
    scenario: str = "sce1",
    list_type: str = "random",
    swap_lists: bool = False,
    list_len: str = "n3",
    prompt_key: int = "1",
    tokenizer_name: str = "gpt2",
    pretokenize_moses: bool = False,
):
    paths = get_paths()

    logger.info(
        f"Creating input sequences for:\n{json.dumps({'condition': condition, 'scenario': scenario, 'list_len': list_len, 'prompt_key': prompt_key}, indent=4)}"
    )

    # ===== DATA MANAGEMENT ===== #

    if list_type == "random":
        input_filename = "random_lists.json"
    elif list_type == "categorized":
        input_filename = "categorized_lists.json"

    # load the word lists in .json files
    fname = os.path.join(paths.data, "noun_lists", input_filename)

    word_lists1, word_lists2 = make_word_lists(fname, condition=condition)

    # if needed, swap lists (useful for patching analyses)
    if swap_lists:
        word_lists2 = word_lists1
        _, word_lists1 = make_word_lists(fname, condition=condition)

    # ===== INITIATE EXPERIMENT CLASS ===== #

    # setup the model
    logger.info("Loading tokenizer {}".format(tokenizer_name))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # ===== CONCATENATE AND TOKENIZE INPUT SEQUENCES ===== # this
    logger.info("Tokenizing and concatenating sequences...")
    input_sequences = concat_and_tokenize_inputs(
        prompt=prompts[scenario][prompt_key],
        prefix=prefixes[scenario]["1"],
        word_list1=word_lists1[list_len],
        word_list2=word_lists2[list_len],
        ngram_size=list_len.strip("n"),
        pretokenize_moses=pretokenize_moses,
        tokenizer=tokenizer,
    )

    # add information about prompt and list lengths for each sequence
    for s in input_sequences:
        s.prompt = prompt_key
    # meta_data["prompt"] = [prompt_key for _ in meta_data["list_len"]]

    return input_sequences


first_sequence_nouns = {
    "n1": "Ġpatience",
    "n2": "Ġnotion",
    "n3": "Ġmovie",
    "n4": "Ġwomen",
    "n5": "Ġcanoe",
    "n6": "Ġnovel",
    "n7": "Ġfolly",
    "n8": "Ġsilver",
    "n9": "Ġeagle",
    "n10": "Ġcenter",
}


def get_query_target_indices(list_len: str, which: str) -> Tuple:
    """A helper function to get the indices of the query and target
    words in the input sequence.
    """

    seqs = get_input_sequences(
        condition="repeat",
        scenario="sce1",
        list_type="random",
        list_len=list_len,
        prompt_key="1",
        tokenizer_name="gpt2",
        pretokenize_moses=False,
    )

    # define indices based on tokens in the first sequences (has no
    # BPE-split tokens)
    t = np.array(seqs[0].toks)

    nouns = list(first_sequence_nouns.values())[0 : int(list_len[-1])]
    codes = list(first_sequence_nouns.keys())[0 : int(list_len[-1])]

    if which == "match":
        query_ids = {c: (n, np.where(t == n)[0][-1]) for n, c in zip(nouns, codes)}
        target_ids = {c: (n, np.where(t == n)[0][0]) for n, c in zip(nouns, codes)}

    elif which == "postmatch":
        increment = 1
        query_ids = {c: (n, np.where(t == n)[0][-1]) for n, c in zip(nouns, codes)}
        target_ids = {
            c: (
                t[(np.where(t == n)[0][0] + increment)],
                (np.where(t == n)[0][0]) + increment,
            )
            for n, c in zip(nouns, codes)
        }

    elif which == "recent":
        increment = -1
        query_ids = {c: (n, np.where(t == n)[0][-1]) for n, c in zip(nouns, codes)}
        target_ids = {
            c: (
                t[np.where(t == n)[0][0] + increment],
                (np.where(t == n)[0][-1] + increment),
            )
            for n, c in zip(nouns, codes)
        }

    return {"queries": query_ids, "targets": target_ids}


def get_inputs_targets_path_patching(batch_size: int = 1):
    inps1 = get_input_sequences(
        condition="repeat",
        scenario="sce1",
        list_type="random",
        list_len="n3",
        prompt_key="1",
    )

    # create corrupted run, by swapping the query and target words
    inps2 = get_input_sequences(
        condition="control",
        swap_lists=True,
        scenario="sce1",
        list_type="random",
        list_len="n3",
        prompt_key="1",
    )

    indices = get_query_target_indices(list_len="n3", which="match")
    colon_at_unsplit = indices["queries"]["n1"][-1] - 1

    second_colon_idx = lambda x: np.where(np.array(x) == ":")[0][-1]

    orig_inps_ids = torch.tensor(
        [
            i
            for i, inps in enumerate(inps1)
            if second_colon_idx(inps.toks) == colon_at_unsplit
        ]
    )  # clean
    corr_inps_ids = torch.tensor(
        [
            i
            for i, inps in enumerate(inps2)
            if second_colon_idx(inps.toks) == colon_at_unsplit
        ]
    )  # corrupted

    # chunk inputs at second colon and stack into single tensor
    orig_inps = torch.stack(
        [inps1[i].ids[0][0 : colon_at_unsplit + 1] for i in orig_inps_ids]
    )
    corr_inps = torch.stack(
        [inps2[i].ids[0][0 : colon_at_unsplit + 1] for i in corr_inps_ids]
    )

    nb = len(orig_inps_ids) // batch_size
    resid = len(orig_inps_ids) % batch_size
    print(nb)
    clean_inps_batches = {
        i // batch_size: orig_inps[i : i + batch_size, :]
        for i in range(0, nb * batch_size, batch_size)
    }
    corr_inps_batches = {
        i // batch_size: corr_inps[i : i + batch_size, :]
        for i in range(0, nb * batch_size, batch_size)
    }

    if resid > 0:
        clean_inps_batches[nb] = orig_inps[-resid:, :]  # zero-indexing
        corr_inps_batches[nb] = corr_inps[-resid:, :]

    # to construct correct/incorrect target pairs, use the indices of
    # the first token in first list (the one we expect to be
    # predicted)
    targets = {
        i // batch_size: torch.tensor(
            [
                [ids1[14].item(), ids2[14].item()]
                for ids1, ids2 in zip(
                    orig_inps[i : i + batch_size], corr_inps[i : i + batch_size]
                )
            ]
        )
        for i in range(0, nb * batch_size, batch_size)
    }

    if resid > 0:
        targets[nb] = torch.tensor(
            [
                [ids1[14], ids2[14]]
                for ids1, ids2 in zip(orig_inps[-resid:, :], corr_inps[-resid:, :])
            ]
        )  # zero-indexing

    return (clean_inps_batches, corr_inps_batches), targets


# ===== Setup for gpt2 ====== #
def get_args_for_dev(
    setup=False,
    scenario="sce1",
    prompt_key="1",
    list_len="n5",
    condition="repeat",
    path_to_tokenizer="gpt2",
    device="cuda",
    input_filename="random_lists.json",
    output_dir=None,
    output_filename=None,
):
    args = {
        "setup": setup,
        "scenario": scenario,
        "prompt_key": prompt_key,
        "list_len": list_len,
        "condition": condition,
        "path_to_tokenizer": path_to_tokenizer,
        "device": device,
        "input_filename": input_filename,
        "output_dir": output_dir,
        "output_filename": output_filename,
    }

    return args


def main():
    sys.path.append("/home/ka2773/project/lm-mem/src/data/")

    # ===== INITIATIONS ===== #

    # collect input arguments
    parser = argparse.ArgumentParser(
        description="surprisal.py runs perplexity experiment"
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="downloads and places nltk model and Tokenizer",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
        help="str, which scenario to use",
    )
    parser.add_argument("--prompt_key", type=str, choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--list_len", type=str, choices=["n3", "n5", "n7", "n10"])
    parser.add_argument(
        "--condition",
        type=str,
        choices=["repeat", "permute", "control"],
        help="str, 'permute' or 'repeat'; whether or not to permute the second word list",
    )
    # To download a different model look at https://huggingface.co/models?filter=gpt2
    parser.add_argument(
        "--pretokenize_moses",
        action="store_true",
        help="Whether or not to pretokenize the input text with MosesTokenizer('en')",
    )
    parser.add_argument(
        "--path_to_tokenizer",
        type=str,
        default="gpt2",
        help="the path to tokenizer folder (expected to work with tokenizer.from_pretrained() method",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="whether to run on cpu or cuda",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        help="str, the name of the .json file containing word lists",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="str, the name of folder to write the output_filename in",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        help="str, the name of the output file saving the dataframe",
    )

    argins = parser.parse_args()

    # if needed for development create the argins namespace with some default values
    # argins = SimpleNamespace(**get_args_for_dev(path_to_tokenizer="transfo-xl-wt103",
    #                                            output_dir="data/transformer_input_files",
    #                                            output_filename="transfo-xl-wt103"))

    # construct output file name and check that it existst
    print(argins.output_dir)
    assert os.path.isdir(argins.output_dir)  # check that the folder exists
    # outpath = os.path.join(savedir, argins.output_filename)

    logger.info("condition == {}".format(argins.condition))
    logger.info("scenario == {}".format(argins.scenario))
    logger.info("list_len == {}".format(argins.list_len))
    logger.info("prompt_key == {}".format(argins.prompt_key))

    # ===== DATA MANAGEMENT ===== #

    # load the word lists in .json files
    fname = os.path.join(DATA_PATH, argins.input_filename)

    word_lists1, word_lists2 = make_word_lists(fname, condition=argins.condition)

    # ===== INITIATE EXPERIMENT CLASS ===== #

    # setup the model
    logger.info("Loading tokenizer {}".format(argins.path_to_tokenizer))
    tokenizer = AutoTokenizer.from_pretrained(argins.path_to_tokenizer)

    # set the flag for function below
    ismlm = False
    if argins.path_to_tokenizer in ["bert-base-uncased"]:
        ismlm = True

    # ===== CONCATENATE AND TOKENIZE INPUT SEQUENCES ===== #

    # this tells the bpe split counter what symbol to look for and how
    # it codes for splits
    bpe_split_marker_dict = {
        "gpt2": "Ġ",
        "/home/ka2773/project/lm-mem/data/wikitext-103_tokenizer": "Ġ",
        "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer": "Ġ",
        "bert-base-uncased": "##",
        "transfo-xl-wt103": None,
    }

    # this tells the bpe split counter how these symbols are used
    marker_logic_dict = {
        "gpt2": "outside",
        "/home/ka2773/project/lm-mem/data/wikitext-103_tokenizer": "outside",
        "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer": "outside",
        "bert-base-uncased": "within",
        "transfo-xl-wt103": None,
    }

    # this routing loops over prompts and prefixes
    # it keeps track of that in meta_data
    logger.info("Tokenizing and concatenating sequences...")
    input_sequences, meta_data = concat_and_tokenize_inputs(
        prompt=prompts[argins.scenario][argins.prompt_key],
        prefix=prefixes[argins.scenario]["1"],
        word_list1=word_lists1[argins.list_len],
        word_list2=word_lists2[argins.list_len],
        ngram_size=argins.list_len.strip("n"),
        pretokenize_moses=argins.pretokenize_moses,
        tokenizer=tokenizer,
    )

    # add information about prompt and list lengths for each sequence
    meta_data["prompt"] = [argins.prompt_key for _ in meta_data["list_len"]]

    # save files
    savename = os.path.join(argins.output_dir, argins.output_filename + ".json")
    logger.info(f"Saving {savename}")
    with open(savename, "w") as fh:
        json.dump([e.tolist() for e in input_sequences], fh, indent=4)

    savename = os.path.join(argins.output_dir, argins.output_filename + "_info.json")
    logger.info(f"Saving {savename}")
    with open(savename, "w") as fh:
        json.dump(meta_data, fh, indent=4)


if __name__ == "__main__":
    main()
