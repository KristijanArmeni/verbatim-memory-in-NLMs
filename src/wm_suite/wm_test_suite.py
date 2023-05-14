
import json
import sys, os
import argparse
from itertools import product
import numpy as np
import numpy.typing as npt
import pandas as pd
import transformers
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, \
                         BertForMaskedLM, TransfoXLLMHeadModel, TransfoXLTokenizer, \
                         AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from typing import List, Dict, Tuple, Any, Sequence
from tqdm import tqdm, trange
import logging
from pprint import pprint

# own modules
sys.path.append("/home/ka2773/project/lm-mem/src")
from data.wt103.dataset import WikiTextDataset
from wm_suite.io.prepare_transformer_inputs import mark_subtoken_splits, make_word_lists, concat_and_tokenize_inputs
from wm_suite.io.stimuli import prefixes, prompts
from wm_suite.wm_ablation import ablate_attn_module

logging.basicConfig(format=("[INFO] %(message)s"), level=logging.INFO)

# own modules
if "win" in sys.platform:
    sys.path.append(os.path.join(os.environ['homepath'], 'project', 'lm-mem', 'src'))
elif "linux" in sys.platform:
    sys.path.append(os.path.join(os.environ['HOME'], 'project', 'lm-mem', 'src'))


# ===== DATASET CLASS ===== #

class SimpleDataset(Dataset):
    def __init__(self, _list: List) -> None:
        super().__init__()
        self.items = list(_list)

    def __getitem__(self, index) -> Any:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)


# ===== HELPER FUNCTION FOR MERGING BPE SPLIT TOKENS ===== #

def code_relative_markers(markers: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to code the position of timesteps relative to lists of nouns within each sequences.
    These markers are used for aggregating prior to visualization.

    Parameters:
    ----------
    markers : np.ndarray or array-like

    Returns:
    tuple of arrays containing the marker positions and relative marker positions
    -------
    """
    marker_pos, marker_pos_rel = [], []

    codes = np.unique(markers)
    for c in codes:

        sel = markers[markers == c]

        if c in [0, 2]:
            # start with 1, reverse and make it negative
            marker_pos_rel.append(-1*(np.arange(0, len(sel))+1)[::-1])
        else:
            marker_pos_rel.append(np.arange(0, len(sel)))

        # code marker position without negative indices
        marker_pos.append(np.arange(0, len(sel)))

    return np.hstack(marker_pos), np.hstack(marker_pos_rel)

def merge_states(x:np.ndarray, bpe_split_indices:np.ndarray, tokens:np.ndarray, mergefun=np.mean):
    """
    finds repeated indices marked by bpe_split_indices() then loops
    and merges over these indices by applying <mergefun> to axis 0
    """

    # find which tokens where split by looking at indices that occur more than once (i.e. for each BPE subword)
    # example: [0, 1, 2, 3, 3, 4] --> token 3 was split into two subwords
    u, c = np.unique(bpe_split_indices, return_counts=True)

    tokens_orig = np.array(tokens)
    new_tokens = np.array(tokens)

    # loop over tokens with count > 1 (these are subword tokens)
    is_repeated = c > 1
    is_not_punct = u > 0
    for wassplit in u[is_repeated & is_not_punct] :

        sel = np.where(bpe_split_indices == wassplit)[0]

        # safety check, merged sub-words must be adjacent, else don't merge
        if (np.diff(sel) == 1).all():

            logging.info(f"Merging tokens {new_tokens[sel]} at positions {sel}")

            # replace first subword token by the mean of all subwords and delete the rest
            x[sel[0], ...] = mergefun(x[sel, ...], axis=0)    # shape = (tokens, ...)
            x[sel[1::], ...] = np.nan                         # 

            # update the strings
            merged_token = "_".join(new_tokens[sel].tolist())

            new_tokens[sel[0]] = merged_token
            new_tokens[sel[1::]] = ''

    return x, new_tokens, tokens_orig

# ===== EXPERIMENT CLASS ===== #

class Experiment(object):

    """
    Exp() class contains wrapper methods to run experiments with transformer models.

    Attributes:
    ----------
    model (required) : nn.Module
        pytorch module representing the model
    ismlm (optional) : boolean
        whether the model is a masked language model or not (default = False)
    device : str
        'cuda' or 'cpu' what compute device to use for running the models
    tokenizer (required) : PreTrainedTokenizer()
        tokenizer class from HuggingFace (https://huggingface.co/transformers/v4.6.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer) 
    context_len : int
        the length of context window for computing transformer surprisal (default = 1024)
    batch_size : int
        the number of sequences evaluated in parallel
    use_cache : bool
        whether or not to reuse key-value matrices from previous time-steps in transformers


    Methods:
    -------


    """

    def __init__(self, model, ismlm, tokenizer, context_len, batch_size, stride, use_cache, device):

        self.model = model
        self.ismlm = ismlm
        self.device = device
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.stride = stride
        self.batch_size = batch_size
        self.use_cache = use_cache

        self.loss_fct_batched = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.to(device)
        

    # loop over input list
    def ppl(self, 
            input_ids: torch.Tensor, 
            context_len: int, 
            stride: int) -> Tuple[List[float], List[float], List[str]]:

        """
        method for computing token-by-token negative log likelihood on input_ids
        taken from: https://huggingface.co/transformers/perplexity.html

        Parameters:
        ----------
        input_ids (required) : torch.tensor
            indices representing tokens to be fed as input to model
        context_len (required) : int
            the length (in number of tokens) of past tokens used for computing negative log likelihood
        stride (required) : int
            the step (in number of tokens) in between two subsequent loops for computing perplexity

        Returns:
        ------
        ppl : 
        llh :
        tokens : 

        """

        llh = []  # variable storing token-by-token neg ll
        tokens = []   # variable storing token strings to have along with -ll in the output

        # loop over tokens in input sequence
        for i in trange(0, input_ids.size(1), stride, desc="Computing perplexity: "):

            # define the start and endpoints of input indices for current loop
            begin_loc = max(i + stride - context_len, 0)  # define the non-negative onset location
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # select the current input index span
            sel_input_ids = input_ids[:, begin_loc: end_loc].clone().to(self.device)

            # mask the final token in sequence if we're testing bert, so that we get prediction only for that token
            if self.ismlm:
                sel_input_ids[0, -1] = self.tokenizer.convert_tokens_to_ids("[MASK]")

            # use unmasked tokens as targets for loss if provided
            target_ids = input_ids[:, begin_loc: end_loc].to(self.device).clone()

            # do not compute the loss on  tokens (-100) that are used for context
            target_ids[:, :-trg_len] = -100

            #if i in range(0, 5):
            #    print(f"inputs: {sel_input_ids}")
            #    print(f"targets: {target_ids}")

            # set model to evaluation mode
            self.model.eval()

            # get model output
            with torch.no_grad():

               # compute neg log likelihood over target ids (n+1 in our case)
               # indices are shifted under the hood by model.__call__()
               outputs = self.model(sel_input_ids, labels=target_ids)
               
               log_likelihood = outputs.loss.item() * trg_len  # not sure about this multiplication here (undoing averaging?)

               llh.append(log_likelihood)
               if stride == 1:
                    toks = self.tokenizer.decode(target_ids[0][-stride::])
                    tokens.append(toks)  # store the last token (target_id)

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / (end_loc-1)).cpu()
        return ppl, llh, tokens

    def ppl_batched(self, 
        input_ids: torch.Tensor, 
        stride: int,
        context_len: int,
        seq_lens: List[int], 
        targets: torch.Tensor) -> Tuple[float, List[float], List[float]]:

        """Returns average ppl, list of suprisal values (llhs)
        per token, and list of tokens for given input_ids.
        taken from: https://huggingface.co/transformers/perplexity.html
        """

        batch_size = len(seq_lens)

        assert batch_size == self.batch_size

        if (stride > 1):
            raise ValueError(".self_batched is only tested on stride == 1,"
                             "higher values are not accepted currently")

        # to match every token need to insert <eos> token artificially at beginning
        llhs = [
            [
                np.nan,
            ]
            for _ in range(batch_size)
        ]  # variable storing token-by-token neg log likelihoods

        # set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)

        if self.use_cache:
            end_loop_idx = (input_ids.size(1) -1)
            past_key_values = None

            # if we're (re)using cache, then we can only select current input slice
            # in this case stride and context_len are ignored
            beg_loc_fun = lambda i, stride=None, context_len=None: i
            end_loc_fun = lambda i, stride=None, max_loc=None: i + 1
            tgt_beg_loc_fun = lambda i, stride=None, context_len=None: i + 1
            tgt_end_loc_fun = lambda i, stride=None, context_len=None: i + 2

        else:
            past_key_values = None
            end_loop_idx = input_ids.size(1) - 1
            beg_loc_fun = lambda i, stride, context_len: max(i + stride - context_len, 0)
            end_loc_fun = lambda i, stride, max_loc: min(i + stride, max_loc)

            # we have to shift targets for one time step forward
            tgt_beg_loc_fun = lambda i, stride, context_len: max((i + stride - context_len), 0) + 1
            tgt_end_loc_fun = lambda i, stride, max_loc: min(i + stride, max_loc) + 1


        # loop over tokens in input sequence
        for idx in range(0, end_loop_idx):

            # compute begining and end indices for context tokens
            beg_loc = beg_loc_fun(idx, stride, context_len)
            end_loc = end_loc_fun(idx, stride, end_loop_idx)
            tgt_beg_loc = tgt_beg_loc_fun(idx, stride, context_len)
            tgt_end_loc = tgt_end_loc_fun(idx, stride, end_loop_idx)

            trg_len = end_loc - idx  # may be different from stride on last loop

            # select the current input index span
            selected_input_ids = input_ids[:, beg_loc:end_loc].to(self.device)

            # mask the final token in sequence if we're testing MLM, so that we get prediction only for that token
            if self.ismlm:
                selected_input_ids[:, -1] = self.tokenizer.convert_tokens_to_ids("[MASK]")

            # targets are shifted by 1 token forward
            target_ids = targets[:, tgt_beg_loc:tgt_end_loc].to(self.device)
            target_ids[:, :-trg_len] = self.loss_fct_batched.ignore_index


            #print(selected_input_ids)
            #print(target_ids)

            # get model output
            with torch.no_grad():

                # compute logits and use cache if provided
                outputs = self.model(
                    input_ids=selected_input_ids,
                )

                #print(outputs.logits.shape)
                #print(outputs.logits[:, -1, 0:10])
                #print(outputs.logits[:, 0, 0:10])

                if self.use_cache:
                    past_key_values = outputs.past_key_values
                del selected_input_ids

                # compute loss per every sequence in batch shape = (batch, sequence_len)
                losses = self.loss_fct_batched(outputs.logits[:, -1, :].view(-1, outputs.logits.size(-1)), 
                                               target_ids[:, -1].view(-1)
                                                )
                # del outputs
                del target_ids

                for batch_idx, nll_val in enumerate(losses):
                    llhs[batch_idx].append(nll_val.item())

                # Save past attention keys for speedup
                # TODO: cannot handle if past_key_values becomes longer than context_length yet

        # Handle padded sequences
        final_llhs = []
        for batch_idx, llh_vals in enumerate(llhs):
            # Cut them off at appropriate length
            final_llhs.append(llh_vals[: seq_lens[batch_idx]])

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppls = []
        for batch_idx, llh_vals in enumerate(final_llhs):
            ppls.append(
                torch.exp(torch.tensor(np.nansum(llh_vals)) / (len(llh_vals) - 1)).cpu().item()
            )
        return ppls, final_llhs

    def start_sequential(self, input_sequences_ids:List[torch.tensor]=None) -> List:
        """
        Parameters:
        ---------
        input_sequences_ids: list of tensors
        
        Returns:
        -------
        outputs : dict
            with fields: 
                "sequence_ppl": List,
                "surp": List
                "token" List: 

        """

        # output dict
        outputs = {
            "sequence_ppl": [],
            "surp": [],
            "token": [],
            }

        # loop over trials (input sequences)
        for _, input_ids in enumerate(input_sequences_ids):

            # this returns surprisal (neg log ll)
            ppl, surp, toks = self.ppl(input_ids=input_ids.to(self.device),
                                       context_len=self.context_len,
                                       stride=1)

            # store the output tuple and
            outputs["sequence_ppl"].append(ppl)
            outputs["surp"].append(surp)
            outputs["token"].append(toks)

        return outputs


    def get_batch(self, sequence_list: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Converts list of sequences into a padded torch Tensor and its lengths"""

        # get lengths of all sequences
        sequence_lengths = [len(sequence[0]) for sequence in sequence_list]

        batched_sequence = torch.nn.utils.rnn.pad_sequence(
            [sequence[0] for sequence in sequence_list],
            batch_first=True,
            padding_value=self.tokenizer.encode(self.tokenizer.unk_token)[0],
        )
        target_sequence = torch.nn.utils.rnn.pad_sequence(
            [sequence[0] for sequence in sequence_list],
            batch_first=True,
            padding_value=self.loss_fct_batched.ignore_index,
        )
        return batched_sequence, sequence_lengths, target_sequence


    def start_batched(self, input_sequences: List[List[torch.Tensor]]) -> Dict:
        """
        Parameters:
        ----------
        input_sequences : list of sequence lists

        Returns:
        -------
        output_dict : dict
            dict with keys:
            'surp': surprisals for each token in the sequence
            'sequence_ppl': perplexity (in bits) for every sequence
            'token': token strings for each time step (if self.stride == 1)
        """

        # initiate dataset class and data loader
        sequence_dataset = SimpleDataset(input_sequences)

        sequence_loader = DataLoader(sequence_dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=False, 
                                     collate_fn=self.get_batch,
        )

        # 3. go through sequences
        outputs = {
            "sequence_ppl": [],
            "surp": [],
        }

        sequence_iterator = tqdm(sequence_loader, desc="Computing surprisal values", colour="blue")

        for input_ids, sequence_lengths, targets in sequence_iterator:

            ppls, surprisals = self.ppl_batched(input_ids=input_ids,
                                                context_len=self.context_len, 
                                                stride=self.stride,
                                                seq_lens=sequence_lengths, 
                                                targets=targets)

            # store the outputs and
            outputs["sequence_ppl"].extend(ppls)
            outputs["surp"].extend(surprisals)

        # store tokens
        if self.stride == 1:
            outputs["token"] = [self.tokenizer.convert_ids_to_tokens(e[0]) for e in input_sequences]

        return outputs

    def start(self, input_sequences: List[torch.Tensor]) -> Dict[str, List[any]]:
        """
        experiment.start() will loop over prefaces, prompts, and word_lists and run the .ppl() method on them
        
        Returns:
        ========
        outputs = {
            "sequence_ppls": [],
            "surprisals": [],
        }
        """

        #if self.batch_size > 1:
        logging.info(f"Batch size = {self.batch_size}, calling self.start_batched()")
        return self.start_batched(input_sequences)

        #logging.info(f"Batch size = {self.batch_size}, calling self.start_sequential()")
        #return self.start_sequential(input_sequences)

    def merge_bpe_splits(self, outputs: Dict) -> Dict:

        for i in range(len(outputs['surp'])):
        
            s = outputs['surp'][i]
            t = outputs['token'][i]

            r = mark_subtoken_splits(t, 
                                     split_marker='Ġ', 
                                     marker_logic='outside', 
                                     eos_markers=["<|endoftext|>", "<|endoftext|>"])

            x, t1, _ = merge_states(np.expand_dims(np.array(s), axis=1), 
                                    bpe_split_indices=np.array(r), 
                                    tokens=t,
                                    mergefun=np.sum)

            outputs['surp'][i] = np.squeeze(x)    # add merged (summed) surprisal
            outputs['token'][i] = t1  # add merged strings

        return outputs


def permute_qk_weights(model=None, per_head=False, seed=None):

    i=0

    if per_head:
        print("shuffling within attn heads...")
    else:
        print("shuffling across attn heads...")

    # access transformer blocks
    for block in tqdm(model.transformer.h, desc="layer"):

        # make seed different for every layer
        rng = np.random.RandomState(seed+(5*i))

        # .weight is a rect matrix of stacked square matrices
        attn_dim = block.attn.c_attn.weight.shape[0]

        # spliting at dim 1 should result in 3 square matrices
        Q, K, V = block.attn.c_attn.weight.split(split_size=attn_dim, dim=1)

        # get the size of each head by diving the embedding size with
        # the number of layers
        head_size = model.config.n_embd//model.config.n_layer

        qk_shuf = []
        for w in (Q, K):

            if not per_head:

                s = w.shape # store original shape

                #flatten, permute across rows/cols and reshape back
                wtmp = rng.permutation(w.detach().numpy().flatten()).reshape(s)
                qk_shuf.append(torch.tensor(wtmp))

            elif per_head:

                # split attn weights into n_layer x n_head square matrices
                heads_shuf = []

                w_attn_heads = w.split(split_size=head_size, dim=1)

                # permute weights within each head
                for j, head in enumerate(w_attn_heads):

                    # pick different seed for layer and each head
                    rng = np.random.RandomState(seed+(5*i+j))
                    s = head.shape # store original shape

                    # flatten, permute across cols/rows, then reshape
                    wtmp = rng.permutation(head.detach().numpy().flatten()).reshape(s)

                    heads_shuf.append(torch.tensor(wtmp))

                qk_shuf.append(torch.cat(heads_shuf, dim=1))

        new_qkv = torch.nn.Parameter(data=torch.cat(qk_shuf + [V], dim=1),
                                     requires_grad=False)

        block.attn.c_attn.weight = new_qkv

        i += 1

    return model


def outputs2dataframe(colnames: List, arrays: List, metacols: List, metarrays: List) -> Dict:

    experiment_outputs = []

    counter = 1  # counter for trials
    n_sequences = len(arrays[0])

    # make a new single dict with all rows for the df below
    dfrows = {key_arr[0]: key_arr[1] for i, key_arr in enumerate(zip(colnames, arrays))}

    # loop over trials
    for i in trange(0, n_sequences):

        # a dict of lists (row values) for this sequence
        row_values = {key: np.array(dfrows[key][i]) for key in dfrows.keys()}

        # convert the last two elements of the tuple to an array
        dftmp = pd.DataFrame(row_values)

        # now add the constant values for the current sequence rows
        for k, x in zip(metacols, metarrays):
            dftmp[k] = x[i]   # this array just contains a constant for the whole sequence             
        
        dftmp['stimID'] = counter                               # track sequence id

        experiment_outputs.append(dftmp)
        counter += 1

    # put into a single df and save
    output = pd.concat(experiment_outputs)

    return output

# ===== Setup for gpt2 ====== #
def setup():
    logging.info("SETUP: nltk punkt")
    import nltk
    nltk.download("punkt")

    logging.info("SETUP: GPT2 Tokenizer")
    GPT2TokenizerFast.from_pretrained("gpt2")

    logging.info("SETUP: Head Model")
    GPT2LMHeadModel.from_pretrained("gpt2")

    return 0

# ===== RUNTIME CODE WRAPPER ===== #
def runtime_code(input_args: List = None):

    from ast import literal_eval
    from string import punctuation

    # ===== INITIATIONS ===== #
    # collect input arguments
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--setup", action="store_true",
                        help="downloads and places nltk model and Tokenizer")
    parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
                        help="str, which scenario to use")
    parser.add_argument("--condition", type=str, choices=["repeat", "permute", "control"])
    parser.add_argument("--list_len", type=str, choices=["3", "5", "7", "10"])
    parser.add_argument("--prompt_len", type=str, choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--list_type", type=str, choices=["random", "categorized"])
    parser.add_argument("--pretokenize_moses", action="store_true")
    parser.add_argument("--noun_list_file", type=str, help="json file with noun lists")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--context_len", type=int, default=1024,
                        help="length of context window in tokens for transformers")
    parser.add_argument("--model_type", type=str, default="pretrained",
                        help="model label controlling which checkpoint to load")
    parser.add_argument("--ablate_layers", type=str, default="none")
    parser.add_argument("--ablate_heads", type=str, default="none")
    parser.add_argument("--ablate_params", type=str, default="Q,K",
                        help="Attention parameters to be ablate. Specify as comma-separated string.")
    parser.add_argument("--ablation_type", type=str, choices=["zero", "shuffle"], default="zero")
    parser.add_argument("--checkpoint", type=str, default="gpt2",
                        help="the path to folder with pretrained models (expected to work with model.from_pretraiend() method)")
    parser.add_argument("--model_seed", type=int, default=12345,
                        help="seed value to be used in torch.manual_seed() prior to calling GPT2Model()")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        help="whether to run on cpu or cuda")
    parser.add_argument("--model_statedict_filename", type=str, default="")
    parser.add_argument("--output_dir", type=str,
                        help="str, the name of folder to write the output_filename in")
    parser.add_argument("--output_filename", type=str,
                        help="str, the name of the output file saving the dataframe")


    # uncomment these to evaluate code below without having to call the script
    #input_args = [
    #    "--scenario", "sce1",
    #    "--condition", "repeat",
    #    "--list_type", "random",
    #    "--list_len", "3",
    #    "--prompt_len", "1",
    #    "--pretokenize_moses",
    #    "--model_type", "ablation",
    #    "--ablate_layer", "(0,)",
    #    "--ablate_head", "(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)",
    #    "--ablation_type", "zero",
    #    "--checkpoint", "gpt2",
    #    "--tokenizer", "gpt2",
    #    #"--checkpoint", "/scratch/ka2773/project/lm-mem/checkpoints/gpt2_full_12-768-1024_a_01/checkpoint-27500",
    #    #"--tokenizer", "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer",
    #    "--model_seed", "12345",
    #    "--output_dir", "/scratch/ka2773/project/lm-mem/output/ablation",
    #    "--output_file", "surprisal_gpt2_pretrained_sce1_repeat_random.csv",
    #    "--device", "cuda",
    #]

    if input_args:
        logging.info("Using input args provided to main(), not from script.")
        argins = parser.parse_args(input_args)
    else:
        argins = parser.parse_args()

    if argins.setup:
        setup()

    sys.path.append("/home/ka2773/project/lm-mem/src/data/")


    # ===== INITIATE MODEL AND TOKANIZER CLASS ===== #

    # declare device and paths
    device = torch.device(argins.device if torch.cuda.is_available() else "cpu")

    # setup the model
    logging.info("Loading tokenizer {}".format(argins.tokenizer))
    tokenizer = GPT2TokenizerFast.from_pretrained(argins.tokenizer)

    # pretrained models
    logging.info("Using {} model".format(argins.model_type))
    logging.info("Loading checkpoint {}".format(argins.checkpoint))
    model = GPT2LMHeadModel.from_pretrained(argins.checkpoint)

    ismlm = False
    if argins.checkpoint == "bert-base-uncased":
        model = BertForMaskedLM.from_pretrained(argins.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(argins.tokenizer)
        ismlm = True

    elif argins.checkpoint == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    elif argins.checkpoint == "transfo-xl-wt103":
        model = TransfoXLLMHeadModel.from_pretrained(argins.checkpoint)
        tokenizer = TransfoXLTokenizer.from_pretrained(argins.tokenizer)

    # or initialize a random model
    if argins.model_type == "random":
        # initialize with random weights
        torch.manual_seed(argins.model_seed)
        model = GPT2LMHeadModel(config=GPT2Config())

    # permute the weights of gpt-small
    elif argins.model_type == "random-att":

        logging.info("Permuting model attention weights ...\n")
        model = permute_qk_weights(model, per_head=False,
                                   seed=argins.model_seed)

    # permute attenion heads of gpt2 small
    elif argins.model_type == "random-att-per-head":

        logging.info("Permuting Q and K weights ...\n")
        model = permute_qk_weights(model, per_head=True,
                                   seed=argins.model_seed)

    # shuffle embedding vectors of gpt2 small
    elif argins.model_type == "shuff-wpe":

        logging.info("Permuting token positions in wpe...")
        rng = np.random.RandomState(seed=argins.model_seed)

        wpe = model.transformer.wpe.weight # shape = (token_positions, embedding_dim)

        # permutation only permutes across 0 dim (rows=token_positions)
        wpe_shuf = torch.tensor(rng.permutation(wpe.detach().numpy()))
        model.transformer.wpe.weight = torch.nn.Parameter(data=wpe_shuf,
                                                          requires_grad=False)

    elif "ablation" in argins.model_type:

        logging.info(f"Running ablation experiment")
        logging.info(f"Setting attention heads {argins.ablate_heads} in layers {argins.ablate_layers} to 0.")

        # now set the selected head in selected layer to 0
        lay = literal_eval(argins.ablate_layers)
        heads = literal_eval(argins.ablate_heads)

        model = ablate_attn_module(model, 
                                   layers = list(lay),
                                   heads = list(heads),
                                   ablation_type="zero")


    # ===== TOKENIZE AND PREPARE THE VIGNETTES ===== #

    # fname = os.path.join(data_dir, argins.input_filename)
    word_lists1, word_lists2 = make_word_lists(argins.noun_list_file, condition=argins.condition)


    # ===== CONCATENATE AND TOKENIZE INPUT SEQUENCES ===== #

    # this tells the bpe split counter what symbol to look for and how it codes for splits
    bpe_split_marker_dict = {"gpt2": "Ġ",
                             "/home/ka2773/project/lm-mem/data/wikitext-103_tokenizer": "Ġ",
                             "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer": "Ġ",
                             "bert-base-uncased": "##",
                             "transfo-xl-wt103": None}

    # this tells the bpe split counter how these symbols are used
    marker_logic_dict = {"gpt2": "outside",
                         "/home/ka2773/project/lm-mem/data/wikitext-103_tokenizer": "outside",
                         "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer": "outside",
                         "bert-base-uncased": "within",
                         "transfo-xl-wt103": None}

    # this routine loops over prompts and prefixes
    # it keeps track of that in meta_data
    logging.info("Tokenizing and concatenating sequences...")
    input_sequences, input_sequences_info = concat_and_tokenize_inputs(prompt=prompts[argins.scenario][argins.prompt_len],
                                                                       prefix=prefixes[argins.scenario]["1"],
                                                                       word_list1=word_lists1[f"n{argins.list_len}"],
                                                                       word_list2=word_lists2[f"n{argins.list_len}"],
                                                                       ngram_size=str(argins.list_len),
                                                                       pretokenize_moses=argins.pretokenize_moses,
                                                                       tokenizer=tokenizer,
                                                                       bpe_split_marker=bpe_split_marker_dict[argins.tokenizer],
                                                                       marker_logic=marker_logic_dict[argins.tokenizer],
                                                                       ismlm=ismlm)

    input_sequences_info["prompt"] = [argins.prompt_len for _ in input_sequences_info['list_len']]


    # ===== EXPERIMENT ===== #

    model.eval()  # set to evaluation mode

    # initialize experiment class
    experiment = Experiment(model=model, ismlm=ismlm,
                            tokenizer=tokenizer,
                            context_len=argins.context_len,
                            batch_size=argins.batch_size,
                            stride=1,
                            use_cache=False,
                            device=device)

    # run experiment
    output_dict = experiment.start(input_sequences = input_sequences)


    # ===== WT-103 PERPLEXITY ===== #

    test_set_path = '/home/ka2773/project/lm-mem/data/wikitext-103/wiki.test.tokens'
    logging.info(f"Loading {test_set_path}...")
    toks, ids = WikiTextDataset(tokenizer=tokenizer).retokenize_txt(test_set_path)

    #initialize experiment class
    experiment2 = Experiment(model=model, ismlm=ismlm,
                            tokenizer=tokenizer,
                            context_len=argins.context_len,
                            batch_size=argins.batch_size,
                            stride=1,
                            use_cache=False,
                            device=device)
    
    ppl, _, _ = experiment2.ppl(input_ids=torch.tensor([ids]), context_len=1024, stride=256)


    # ===== POST PROCESSING ===== #

    logging.info("Postprocessing outputs...")

    # merge tokens and surprisals that were splits into BPE tokens
    output_dict = experiment.merge_bpe_splits(output_dict)

    # find timesteps that correspond to punctuation, end of string and empty slots
    punct_symbols = [np.array([s.strip("Ġ") in punctuation for s in l]) for l in output_dict['token']]        # find positions of punctuation symbols
    eos_symbols = [np.array([s.strip("Ġ") in tokenizer.eos_token for s in l]) for l in output_dict['token']]  # find positions for end of text symbols
    empty_symbols = [np.array([s.strip("Ġ") == '' for s in l]) for l in output_dict['token']]                 # make sure to track empty strings after merging
    
    # create a joint boolean, per sequence
    timesteps = [[(~punct & ~eos & ~empty) for punct, eos, empty in zip(x, y, z)] for x, y, z in zip(punct_symbols, eos_symbols, empty_symbols)]

    # create nan arrays that are going to be populated with markers and will have nan values elsewhere
    output_dict['marker_pos'] = [np.full(shape=len(s), fill_value=np.nan) for s in output_dict['token']]
    output_dict['marker_pos_rel'] = [np.full(shape=len(s), fill_value=np.nan) for s in output_dict['token']]

    # now create absolute and relative markers for each sequence
    for i, markers in enumerate(input_sequences_info['trialID']):

        sel = timesteps[i]
        output_dict['marker_pos'][i][sel] = code_relative_markers(np.array(markers)[sel])[0]
        output_dict['marker_pos_rel'][i][sel] = code_relative_markers(np.array(markers)[sel])[1]

    experiment.model.to("cpu")

    # ===== SAVING MODEL AND PERPLEXITY ===== #

    if argins.model_statedict_filename:

        fn = os.path.join(argins.output_dir, argins.model_statedict_filename)
        logging.info(f"Saving {fn}")
        torch.save(experiment.model.state_dict(), fn)

            # store wikitext-103 perplexity
        ppl_dict = {"wt103_ppl": round(ppl.cpu().item(), 2),
                    "model": argins.model_statedict_filename}
        
        fn = os.path.join(argins.output_dir, argins.model_statedict_filename.replace(".pt", "_ppl.json"))
        with open(fn, 'w') as fh:
            json.dump(ppl_dict, fh)

    # ===== FORMAT AND SAVE OUTPUT FILES ===== #

    colnames = ["token", "surp", "marker_pos", "marker_pos_rel"]
    metacols = ["ppl", "scenario", "second_list", "list", "trialID", "positionID", "subtok", "list_len", "prompt_len"]

    arrays = [output_dict["token"],
              output_dict["surp"],
              output_dict['marker_pos'],
              output_dict['marker_pos_rel']]
    
    metarrays = [output_dict['sequence_ppl'],                                                  
                 [argins.scenario for _ in range(len(input_sequences_info["trialID"]))],  # "sce1" | "sce2" | etc. 
                 [argins.condition for _ in range(len(input_sequences_info["trialID"]))], # "repeat" | "permute" | "control"
                 [argins.list_type for _ in range(len(input_sequences_info["trialID"]))], # "random" | "categorized"
                 input_sequences_info["trialID"],
                 input_sequences_info["positionID"],
                 input_sequences_info["subtok"],
                 input_sequences_info["list_len"],
                 input_sequences_info["prompt"]]

    # put everything into a dataframe
    logging.info("Converting to dataframe...")

    output_df = outputs2dataframe(colnames, arrays, metacols, metarrays)

    output_df.attrs = {'wt103_ppl': np.round(ppl.cpu().item(), 2),
                       'argins': argins}

    # drop nan rows (merged BPE timesteps)
    output_df = output_df.loc[~output_df.token.str.strip("Ġ").isin([""]), :]

    # save output
    outpath = os.path.join(argins.output_dir, argins.output_filename)
    logging.info("Saving {}".format(os.path.join(outpath)))
    output_df.to_csv(outpath, sep="\t")

# ===== RUN ===== #
if __name__ == "__main__":

    runtime_code()
