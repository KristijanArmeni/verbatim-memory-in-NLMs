import os
import json
from wm_suite.io.prepare_transformer_inputs import mark_subtoken_splits, make_word_lists, concat_and_tokenize_inputs
from wm_suite.io.stimuli import prefixes, prompts
from wm_suite.wm_test_suite import merge_states
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import numpy as np
import logging


def load_json_file(fname: str) -> Dict:

    logging.info(f"Loading {fname}")
    with open(fname, 'r') as fh:
        d = json.load(fh)

    return d


def get_attention_weights(model: GPT2LMHeadModel, sequences: List[int], query_tokens_ids: List[int], device: str) -> List[torch.Tensor]:
    """
    get_attention_weigts() loops over <sequences> and extracts attention weights (o.attentions field in GPT2LMHeadModel output)

    Parameters
    ----------
    model : transformers.GPT2LMHeadModel
        The model to extract attention weights from.
    sequences : List[List[int]]
        tensors with tokens converted to indices fed as inputs to model
    query_token_ids : List[int]
        for every sequence in <sequences>, it contains an int indexing the position of the query token (the token model is attending from)

    Returns
    -------
    list
        list of tensors (shape = (layers, heads, sequence_len)) conatining attention weights from query token to past context in sequence.
    """


    # loop over sequences and extract attention weights
    attn, labs = [], []
    with torch.no_grad():

        for inputs, from_token in tqdm(zip(sequences, query_tokens_ids), desc="sequence:", total=len(sequences)):

            o = model(torch.LongTensor(inputs).to(device), 
                      output_hidden_states=False,
                      output_attentions=True)

            # loop over layers and store only the row in attention matrix that has attention from <from_token> to beginning
            attns = [layer[:, :, from_token, :] for layer in o.attentions] # layer.shape = (batch_size, n. heads, sequence_len, sequence_len)

            # concatenate layer dimension across dimension 0
            attn.append(torch.cat(attns, dim=0)) # shape = (layers, heads, sequence_len)
            labs.append(["layer", "head", "seqlen"])


    return attn


def find_cue_token_ids(markers: np.ndarray) -> Tuple[int, int]:
    """"
    Parameters:
    ----------
    markers : np.array
        array with ints coding prefix (0), first list (1), intervening text (2) or second list (3).

    Returns:
    -------
    a : int
    b : int
    """
    prefix_code = 0
    intervening_text_code = 2
    a = np.where(markers == prefix_code)[0][-1]            # find the final token in the prefix (coded as 1) (":" etc.)
    b = np.where(markers == intervening_text_code)[0][-1]  # find the final token in the intervening text (coded as 2)

    return a, b


def find_list_punctuation_index(markers, subtok_markers):
    """
    Parameters
    ----------
    markers : np.array
        array containing indices marking subparts of the string (index 3 indicates second list)
    subtok_markers : np.array
        array containing indices marking whether or not a token was split into BPE subwords (-1 markes punctuation)

    Returns
    -------
    a : np.array
        array containing indices of punctuation symbols in the second list
    """
    second_list_marker = 3
    punctuation_marker = -1
    a = np.where((markers == second_list_marker) & (subtok_markers == punctuation_marker))[0]

    return a


def replace_cue_symbols(sequences: List, tokens: List, markers: List, symbols: Tuple[str, str], tokenizer) -> Tuple[List, List]:
    """
    A helper function that loops over all the lists in the experiment, finds indices of cue symbols (symbols that
    introduce list1 and list2) and replaces them with tokens stored in <symbols>
    
    Parameters
    ----------
    symbols : Tuple
        a tuple, containing tokens introducing the first and the second list respectively
    
    Returns
    -------
    output_tuple : Tuple[List, List]
    """
    cue_token1, cue_token2 = symbols
    idx1 = tokenizer.convert_tokens_to_ids(cue_token1)
    idx2 = tokenizer.convert_tokens_to_ids(cue_token2)

    for j in range(len(sequences)):

        # extract indices marking the position of cue symbols
        ons1, ons2 = find_cue_token_ids(np.array(markers[j]))
        
        # if any of the symbols is to be replaced, do it
        if cue_token1 != "":
            sequences[j][0][ons1] = idx1
            tokens[j][ons1] = cue_token1
        
        if cue_token2 != "":
            sequences[j][0][ons2] = idx2
            tokens[j][ons2] = cue_token2

    return sequences, tokens


def get_args_for_devtesting():

    input_args = ["--scenario", "sce1",
                  "--condition", "repeat",
                  "--list_type", "random",
                  "--noun_list_file", "/home/ka2773/project/lm-mem/src/data/noun_lists/random_lists.json",
                  "--list_len", "10",
                  "--prompt_len", "1",
                  "--query_token", "c5",
                  "--list1_cue_symbol", "",
                  "--list2_cue_symbol", "",
                  "--savedir", "/scratch/ka2773/project/lm-ecog/data/attn",
                  "--checkpoint", "gpt2",
                  "--tokenizer", "gpt2",
                  #"--state_dict", "/scratch/ka2773/project/lm-mem/output/ablation/gpt2_ablate-11-all.pt",
                  "--output_filename", "gpt2_attn_test.npz"]
                  
    return input_args

def main(input_args=None, devtesting=False):

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
                        help="str, which scenario to use")
    parser.add_argument("--condition", type=str, choices=["repeat", "permute", "control"])
    parser.add_argument("--list_len", type=str, choices=["3", "5", "7", "10"])
    parser.add_argument("--prompt_len", type=str, choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--list_type", type=str, choices=["random", "categorized"])
    parser.add_argument("--noun_list_file", type=str, help="json file with noun lists")
    parser.add_argument("--pretokenize_moses", action="store_true")
    parser.add_argument("--list1_cue_symbol", type=str, default='',
                        help="[optional] Symbol that introduces first list.")
    parser.add_argument("--list2_cue_symbol", type=str, default='',
                        help="[optional] Symbol that introduces second list.")
    parser.add_argument("--query_token", type=str,
                        help="p1 == prompt token (e.g. ':'), n1 == first token in the repeated list, n2 == second token in the repeated list etc.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--state_dict", type=str, default="")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--output_filename", type=str)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if devtesting:
        input_args = get_args_for_devtesting()

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    logging.info(f"Using the config\n:{json.dumps(vars(args), indent=2)}")

    if args.checkpoint:
        # load the model checkpoint
        try:
            assert args.state_dict == ""
        except:
            raise ValueError("Can't have args.checkpoint and args.state_dict")
            
        model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
        
    if args.state_dict:  # for loading ablation models
                # load the model checkpoint
        try:
            assert (args.checkpoint == "")
        except:
            raise ValueError("Can't have args.checkpoint and args.state_dict")

        logging.info(f"Loading statedict {args.state_dict}")

        model_unablated = GPT2LMHeadModel.from_pretrained("gpt2")
        configuration = model_unablated.config                     # find the config of the pretrained model
        model = GPT2LMHeadModel(configuration)                     # use that config to initialized class for ablated model
        model.load_state_dict(torch.load(args.state_dict))

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)


    # ===== GENERATE, CONCATENATE AND TOKENIZE INPUT SEQUENCES ===== #

    # fname = os.path.join(data_dir, argins.input_filename)
    word_lists1, word_lists2 = make_word_lists(args.noun_list_file, condition=args.condition)

    # this tells the bpe split counter what symbol to look for and how it codes for splits
    bpe_split_marker_dict = {"gpt2": "Ġ",
                             "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer": "Ġ"}

    # this tells the bpe split counter how these symbols are used
    marker_logic_dict = {"gpt2": "outside",
                         "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer": "outside"}

    # this routing loops over prompts and prefixes
    # it keeps track of that in meta_data
    logging.info("Tokenizing and concatenating sequences...")
    input_sequences, input_sequences_info = concat_and_tokenize_inputs(prompt=prompts[args.scenario][args.prompt_len],
                                                                       prefix=prefixes[args.scenario]["1"],
                                                                       word_list1=word_lists1[f"n{args.list_len}"],
                                                                       word_list2=word_lists2[f"n{args.list_len}"],
                                                                       ngram_size=str(args.list_len),
                                                                       pretokenize_moses=args.pretokenize_moses,
                                                                       tokenizer=tokenizer,
                                                                       bpe_split_marker=bpe_split_marker_dict[args.tokenizer],
                                                                       marker_logic=marker_logic_dict[args.tokenizer],
                                                                       ismlm=False)

    # add prompt field to the dict
    input_sequences_info["prompt"] = [args.prompt_len for _ in input_sequences_info['list_len']]

    # convert ids to strings to have them at hand for plotting
    strings = [tokenizer.convert_ids_to_tokens(e[0]) for e in input_sequences]
    
    # find the query token index (the index can vary if there's BPE splits in the string)
    if args.query_token == "p1":
        
        query_idxs = [find_cue_token_ids(np.array(markers))[1] for markers in input_sequences_info["trialID"]]

    elif args.query_token in ["r1", "r2", "r3", "r4"]:

        tag2id = {"r1": 1, "r2": 2, "r3": 3, "r4": 4}
        offset_from_comma = tag2id[args.query_token]   # get number of positions to subtract from comma position
        query_idxs = [find_cue_token_ids(np.array(markers))[1] - offset_from_comma for markers in input_sequences_info["trialID"]]

        # the nouns we should find at these positions are known, make a quick check
        if args.query_token == "r1":
            assert sum([strings[i][idx] == "Ġagain" for i, idx in enumerate(query_idxs)]) == 230
        elif args.query_token == "r2":
            assert sum([strings[i][idx] == "Ġlist" for i, idx in enumerate(query_idxs)]) == 230
        elif args.query_token == "r3":
            assert sum([strings[i][idx] == "Ġthe" for i, idx in enumerate(query_idxs)]) == 230
        elif args.query_token == "r4":
            assert sum([strings[i][idx] == "Ġread" for i, idx in enumerate(query_idxs)]) == 230

    # if the query token is a noun in the list, find it by looking for list marker and puncutation markers
    elif args.query_token in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10']:
        
        tag2id = {f'n{i+1}': i for i in range(10)} # index to pick the right id
        punct_id = tag2id[args.query_token]  # grab id for the specified query noun
        index_offset = -1  # shift by one position so we get the nouns before the commas
        query_idxs = [find_list_punctuation_index(np.array(markers), np.array(punct_markers))[punct_id] - index_offset for markers, punct_markers in zip(input_sequences_info["trialID"], input_sequences_info["subtok"])]

    elif args.query_token in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']:
        
        tag2id = {f'c{i+1}': i for i in range(9)}
        punct_id = tag2id[args.query_token]  # grab id for the specified query comma
        query_idxs = [find_list_punctuation_index(np.array(markers), np.array(punct_markers))[punct_id] for markers, punct_markers in zip(input_sequences_info["trialID"], input_sequences_info["subtok"])]

        # check that we've indeed found the commas at all positions
        assert sum([strings[i][idx] == "," for i, idx in enumerate(query_idxs)])

    # replace ':' (default) by another character is specified (e.g. for control experiments)
    control_cue_tokens = [';', ',']
    if (args.list1_cue_symbol in control_cue_tokens) or (args.list2_cue_symbol in control_cue_tokens):
        
        logging.info("Replacing one (or both) of the list cue symbols...")

        input_sequences, strings = replace_cue_symbols(sequences=input_sequences, 
                                                       tokens=strings, 
                                                       markers=input_sequences_info['trialID'],
                                                       symbols=(args.list1_cue_symbol, args.list2_cue_symbol),
                                                       tokenizer=tokenizer)


    # store attention weights going from query token to context
    r = get_attention_weights(model.to(device), 
                              input_sequences, 
                              query_tokens_ids=query_idxs,
                              device=device)

    # output data structure
    out = {'data': [], 'tokens': [], 'label': []}

    # loop through sequence attention weights and merge the BPE ones
    for d, s in zip(r, strings):

        bpe_splits = mark_subtoken_splits(tokens=s, 
                                          split_marker=bpe_split_marker_dict[args.tokenizer], 
                                          marker_logic=marker_logic_dict[args.tokenizer],
                                          eos_markers=["<|endoftext|>"])

        # move tokens as first dimension and layer as last
        x = d.T.cpu().numpy().copy()  # shape = (tokens, head, layers)
        o = merge_states(x=x, bpe_split_indices=bpe_splits, tokens=s, mergefun=np.sum)
        
        # remove empty (merged) timesteps
        isna = np.where(np.isnan(o[0]))[0]
        out['data'].append(np.delete(o[0], isna, axis=0))
        out['tokens'].append(np.delete(o[1], isna, axis=0))
        out['label'].append(('token', 'heads', 'layers'))

    # save attention weights and analyze offline
    savename = os.path.join(args.savedir, args.output_filename)
    logging.info(f"Saving {savename}")
    np.savez(savename, **out)

    return 0

if __name__ == "__main__":

    main()