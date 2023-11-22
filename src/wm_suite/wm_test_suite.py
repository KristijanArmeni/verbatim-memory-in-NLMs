import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.stats import bootstrap, median_abs_deviation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from wm_suite.io.prepare_transformer_inputs import (
    get_input_sequences,
    mark_subtoken_splits,
)
from wm_suite.io.test_ds import get_test_data
from wm_suite.io.wt103.dataset import WikiTextDataset
from wm_suite.paths import get_paths
from wm_suite.utils import logger
from wm_suite.viz.func import filter_and_aggregate
from wm_suite.wm_ablation import (
    ablate_attn_module,
    find_topk_attn,
    find_topk_intersection,
    from_dict_to_labels,
    from_labels_to_dict,
)


PATHS = get_paths()

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
    """Helper function to code the position of timesteps relative to
    lists of nouns within each sequences.  These markers are used for
    aggregating prior to visualization.

    Parameters
    ----------
    markers : np.ndarray or array-like

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]

    """
    marker_pos, marker_pos_rel = [], []

    codes = np.unique(markers)
    for c in codes:
        sel = markers[markers == c]

        if c in [0, 2]:
            # start with 1, reverse and make it negative
            marker_pos_rel.append(-1 * (np.arange(0, len(sel)) + 1)[::-1])
        else:
            marker_pos_rel.append(np.arange(0, len(sel)))

        # code marker position without negative indices
        marker_pos.append(np.arange(0, len(sel)))

    return np.hstack(marker_pos), np.hstack(marker_pos_rel)


def merge_states(
    x: np.ndarray, bpe_split_indices: np.ndarray, tokens: np.ndarray, mergefun=np.mean
):
    """
    Finds repeated indices marked by bpe_split_indices() then loops
    and merges over these indices by applying <mergefun> to axis 0

    Parameters
    ----------

    Returns
    -------

    """

    # find which tokens where split by looking at indices that occur
    # more than once (i.e. for each BPE subword)
    # example: [0, 1, 2, 3, 3, 4] --> token 3 was split into two subwords
    u, c = np.unique(bpe_split_indices, return_counts=True)

    tokens_orig = np.array(tokens)
    new_tokens = np.array(tokens)

    # loop over tokens with count > 1 (these are subword tokens)
    is_repeated = c > 1
    is_not_punct = u > 0
    for wassplit in u[is_repeated & is_not_punct]:
        sel = np.where(bpe_split_indices == wassplit)[0]

        # safety check, merged sub-words must be adjacent, else don't merge
        if (np.diff(sel) == 1).all():
            logger.info(f"Merging tokens {new_tokens[sel]} at positions {sel}")

            # replace first subword token by the mean of all subwords
            # and delete the rest
            x[sel[0], ...] = mergefun(x[sel, ...], axis=0)  # shape = (tokens, ...)
            x[sel[1::], ...] = np.nan  #

            # update the strings
            merged_token = "_".join(new_tokens[sel].tolist())

            new_tokens[sel[0]] = merged_token
            new_tokens[sel[1::]] = ""

    return x, new_tokens, tokens_orig


# ===== EXPERIMENT CLASS ===== #


class Experiment(object):

    """Experiment() class contains wrapper methods to run experiments
    with transformer models.

    Attributes
    ----------
    model : nn.Module
        pytorch module representing the model
    ismlm : bool
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
        whether or not to reuse key-value matrices from previous
        time-steps in transformers
    loss_fct_batched : instace of `torch.nn.CrossEntropyLoss(reduction="none")`


    Examples
    --------
    ```python
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch

    model = GPT2LMHeadModel.from_pretrained("gpt2")   # load pretrained weights
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    model.eval()  # set to evaluation mode

    # initialize experiment class
    experiment = Experiment(model=model,
                            ismlm=False,
                            tokenizer=tokenizer,
                            context_len=1024,
                            batch_size=1,
                            stride=1,
                            use_cache=False,
                            device="cpu")  # or "cuda"

    # run experiment
    output_dict = experiment.start(input_sequences = input_sequences)
    ```

    """

    def __init__(
        self,
        model,
        ismlm,
        tokenizer,
        context_len,
        batch_size,
        stride,
        use_cache,
        device,
    ):
        """
        Parameters
        ----------
        model (required) : nn.Module
            pytorch module representing the model (is moved to
            self.device upon initialization)
        ismlm (optional) : bool
            whether the model is a masked language model or not (default = False)
        device : str
            'cuda' or 'cpu' what compute device to use for running the models
        tokenizer (required) : PreTrainedTokenizer()
            tokenizer class from HuggingFace (https://huggingface.co/transformers/v4.6.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer)
        context_len : int
            the length of context window for computing transformer
            surprisal (default = 1024)
        batch_size : int
            the number of sequences evaluated in parallel
        use_cache : bool
            whether or not to reuse key-value matrices from previous
            time-steps in transformers

        """

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
    def ppl(
        self, input_ids: torch.Tensor, context_len: int, stride: int
    ) -> Tuple[List[float], List[float], List[str]]:
        """
        Method for computing token-by-token negative log likelihood on input_ids
        taken from: https://huggingface.co/transformers/perplexity.html

        Parameters
        ----------
        input_ids (required) : torch.tensor
            indices representing tokens to be fed as input to model
        context_len (required) : int
            the length (in number of tokens) of past tokens used for
            computing negative log likelihood
        stride (required) : int
            the step (in number of tokens) in between two subsequent
            loops for computing perplexity

        Returns
        -------
        ppl :
        llh :
        tokens :

        """

        llh = []  # variable storing token-by-token neg ll
        # variable storing token strings to have along with -ll in the output
        tokens = []

        # loop over tokens in input sequence
        for i in trange(0, input_ids.size(1), stride, desc="Computing perplexity: "):
            # define the start and endpoints of input indices for current loop
            begin_loc = max(
                i + stride - context_len, 0
            )  # define the non-negative onset location
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # select the current input index span
            sel_input_ids = input_ids[:, begin_loc:end_loc].clone().to(self.device)

            # mask the final token in sequence if we're testing bert,
            # so that we get prediction only for that token
            if self.ismlm:
                sel_input_ids[0, -1] = self.tokenizer.convert_tokens_to_ids("[MASK]")

            # use unmasked tokens as targets for loss if provided
            target_ids = input_ids[:, begin_loc:end_loc].to(self.device).clone()

            # do not compute the loss on  tokens (-100) that are used for context
            target_ids[:, :-trg_len] = -100

            # if i in range(0, 5):
            #    print(f"inputs: {sel_input_ids}")
            #    print(f"targets: {target_ids}")

            # set model to evaluation mode
            self.model.eval()

            # get model output
            with torch.no_grad():
                # compute neg log likelihood over target ids (n+1 in our case)
                # indices are shifted under the hood by model.__call__()
                outputs = self.model(sel_input_ids, labels=target_ids)

                log_likelihood = (
                    outputs.loss.item() * trg_len
                )  # not sure about this multiplication here (undoing averaging?)

                llh.append(log_likelihood)
                if stride == 1:
                    toks = self.tokenizer.decode(target_ids[0][-stride::])
                    tokens.append(toks)  # store the last token (target_id)

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / (end_loc - 1)).cpu()
        return ppl, llh, tokens

    def ppl_batched(
        self,
        input_ids: torch.Tensor,
        stride: int,
        context_len: int,
        seq_lens: List[int],
        targets: torch.Tensor,
    ) -> Tuple[float, List[float], List[float]]:
        """Returns average ppl, list of suprisal values (llhs)
        per token, and list of tokens for given input_ids.
        taken from: https://huggingface.co/transformers/perplexity.html
        """

        batch_size = len(seq_lens)

        assert batch_size == self.batch_size

        if stride > 1:
            raise ValueError(
                ".self_batched is only tested on stride == 1,"
                "higher values are not accepted currently"
            )

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
            end_loop_idx = input_ids.size(1) - 1
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
            beg_loc_fun = lambda i, stride, context_len: max(
                i + stride - context_len, 0
            )
            end_loc_fun = lambda i, stride, max_loc: min(i + stride, max_loc)

            # we have to shift targets for one time step forward
            tgt_beg_loc_fun = (
                lambda i, stride, context_len: max((i + stride - context_len), 0) + 1
            )
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

            # mask the final token in sequence if we're testing MLM,
            # so that we get prediction only for that token
            if self.ismlm:
                selected_input_ids[:, -1] = self.tokenizer.convert_tokens_to_ids(
                    "[MASK]"
                )

            # targets are shifted by 1 token forward
            target_ids = targets[:, tgt_beg_loc:tgt_end_loc].to(self.device)
            target_ids[:, :-trg_len] = self.loss_fct_batched.ignore_index

            # print(selected_input_ids)
            # print(target_ids)

            # get model output
            with torch.no_grad():
                # compute logits and use cache if provided
                outputs = self.model(
                    input_ids=selected_input_ids,
                )

                # print(outputs.logits.shape)
                # print(outputs.logits[:, -1, 0:10])
                # print(outputs.logits[:, 0, 0:10])

                if self.use_cache:
                    past_key_values = outputs.past_key_values
                del selected_input_ids

                # compute loss per every sequence in batch shape = (batch, sequence_len)
                losses = self.loss_fct_batched(
                    outputs.logits[:, -1, :].view(-1, outputs.logits.size(-1)),
                    target_ids[:, -1].view(-1),
                )
                # del outputs
                del target_ids

                for batch_idx, nll_val in enumerate(losses):
                    llhs[batch_idx].append(nll_val.item())

                # Save past attention keys for speedup TODO: cannot
                # handle if past_key_values becomes longer than
                # context_length yet

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
                torch.exp(torch.tensor(np.nansum(llh_vals)) / (len(llh_vals) - 1))
                .cpu()
                .item()
            )
        return ppls, final_llhs

    def get_batch(self, sequence_list: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Converts list of sequences into a padded torch Tensor and its lengths"""

        # get lengths of all sequences
        sequence_lengths = [len(sequence) for sequence in sequence_list]

        batched_sequence = torch.nn.utils.rnn.pad_sequence(
            sequence_list,
            batch_first=True,
            padding_value=self.tokenizer.encode(self.tokenizer.unk_token)[0],
        )
        target_sequence = torch.nn.utils.rnn.pad_sequence(
            sequence_list,
            batch_first=True,
            padding_value=self.loss_fct_batched.ignore_index,
        )
        return batched_sequence, sequence_lengths, target_sequence

    def start_batched(self, input_sequences: List[List[torch.Tensor]]) -> Dict:
        """
        Parameters
        ----------
        input_sequences : List[List[torch.Tensor]]

        Returns
        -------
        dict
            dict with keys:

                - 'surp' surprisals for each token in the sequence
                - 'sequence_ppl': perplexity (in bits) for every sequence
                - 'token': token strings for each time step (if self.stride == 1)
        """

        # initiate dataset class and data loader
        sequence_dataset = SimpleDataset(input_sequences)

        sequence_loader = DataLoader(
            sequence_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.get_batch,
        )

        # 3. go through sequences
        outputs = {
            "sequence_ppl": [],
            "surp": [],
        }

        sequence_iterator = tqdm(
            sequence_loader, desc="Computing surprisal values", colour="blue"
        )

        for input_ids, sequence_lengths, targets in sequence_iterator:
            ppls, surprisals = self.ppl_batched(
                input_ids=input_ids,
                context_len=self.context_len,
                stride=self.stride,
                seq_lens=sequence_lengths,
                targets=targets,
            )

            # store the outputs and
            outputs["sequence_ppl"].extend(ppls)
            outputs["surp"].extend(surprisals)

        # store tokens
        if self.stride == 1:
            outputs["token"] = [
                self.tokenizer.convert_ids_to_tokens(e) for e in input_sequences
            ]

        return outputs

    def start(self, input_sequences: List[torch.Tensor]) -> Dict[str, List[any]]:
        """experiment.start() will call the [wm_suite.Experiment][]
        method on `input_sequences`

        """

        logger.info(f"Batch size = {self.batch_size}, calling self.start_batched()")
        return self.start_batched(input_sequences)

    def merge_bpe_splits(self, outputs: Dict) -> Dict:
        for i in range(len(outputs["surp"])):
            s = outputs["surp"][i]
            t = outputs["token"][i]

            r = mark_subtoken_splits(
                t,
                split_marker="Ġ",
                marker_logic="outside",
                eos_markers=["<|endoftext|>", "<|endoftext|>"],
            )

            x, t1, _ = merge_states(
                np.expand_dims(np.array(s), axis=1),
                bpe_split_indices=np.array(r),
                tokens=t,
                mergefun=np.sum,
            )

            outputs["surp"][i] = np.squeeze(x)  # add merged (summed) surprisal
            outputs["token"][i] = t1  # add merged strings

        return outputs


def permute_qk_weights(model=None, per_head=False, seed=None):
    i = 0

    if per_head:
        print("shuffling within attn heads...")
    else:
        print("shuffling across attn heads...")

    # access transformer blocks
    for block in tqdm(model.transformer.h, desc="layer"):
        # make seed different for every layer
        rng = np.random.RandomState(seed + (5 * i))

        # .weight is a rect matrix of stacked square matrices
        attn_dim = block.attn.c_attn.weight.shape[0]

        # spliting at dim 1 should result in 3 square matrices
        Q, K, V = block.attn.c_attn.weight.split(split_size=attn_dim, dim=1)

        # get the size of each head by diving the embedding size with
        # the number of layers
        head_size = model.config.n_embd // model.config.n_layer

        qk_shuf = []
        for w in (Q, K):
            if not per_head:
                s = w.shape  # store original shape

                # flatten, permute across rows/cols and reshape back
                wtmp = rng.permutation(w.detach().numpy().flatten()).reshape(s)
                qk_shuf.append(torch.tensor(wtmp))

            elif per_head:
                # split attn weights into n_layer x n_head square matrices
                heads_shuf = []

                w_attn_heads = w.split(split_size=head_size, dim=1)

                # permute weights within each head
                for j, head in enumerate(w_attn_heads):
                    # pick different seed for layer and each head
                    rng = np.random.RandomState(seed + (5 * i + j))
                    s = head.shape  # store original shape

                    # flatten, permute across cols/rows, then reshape
                    wtmp = rng.permutation(head.detach().numpy().flatten()).reshape(s)

                    heads_shuf.append(torch.tensor(wtmp))

                qk_shuf.append(torch.cat(heads_shuf, dim=1))

        new_qkv = torch.nn.Parameter(
            data=torch.cat(qk_shuf + [V], dim=1), requires_grad=False
        )

        block.attn.c_attn.weight = new_qkv

        i += 1

    return model


def outputs2dataframe(
    colnames: List, arrays: List, metacols: List, metarrays: List
) -> Dict:
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
            dftmp[k] = x[
                i
            ]  # this array just contains a constant for the whole sequence

        dftmp["stimID"] = counter  # track sequence id

        experiment_outputs.append(dftmp)
        counter += 1

    # put into a single df and save
    output = pd.concat(experiment_outputs, ignore_index=True)

    return output


def get_input_args_for_testing():
    logger.info("Using input args provided to main(), not from script.")

    arglist = [
        "--scenario",
        "sce1",
        "--condition",
        "repeat",
        "--list_type",
        "random",
        "--list_len",
        "3",
        "--prompt_len",
        "1",
        "--model_type",
        "ablated",
        "--model_id",
        "ablate_1-5",
        "--ablate_layer_head_dict",
        "{1: [0, 1, 2], 5: [5, 6]}",
        "--ablation_type",
        "zero",
        "--checkpoint",
        "gpt2",
        "--tokenizer",
        "gpt2",
        "--model_seed",
        "12345",
        "--noun_list_file",
        "/home/ka2773/project/lm-mem/src/data/noun_lists/random_lists.json",
        "--output_dir",
        "./",
        "--output_filename",
        "test.csv",
        "--device",
        "cuda",
        "--output_dir",
        "/home/ka2773/project/lm-mem/test",
    ]

    return arglist


# ===== RUNTIME CODE WRAPPER ===== #
def main(input_args: List = None, devtesting: bool = False):
    from ast import literal_eval
    from string import punctuation

    # ===== INITIATIONS ===== #
    # collect input arguments
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--test_run", action="store_true")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
        help="str, which scenario to use",
    )
    parser.add_argument(
        "--condition", type=str, choices=["repeat", "permute", "control"]
    )
    parser.add_argument("--list_len", type=str, choices=["3", "5", "7", "10"])
    parser.add_argument("--prompt_len", type=str, choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--list_type", type=str, choices=["random", "categorized"])
    parser.add_argument("--pretokenize_moses", action="store_true")
    parser.add_argument("--noun_list_file", type=str, help="json file with noun lists")
    parser.add_argument(
        "--context_len",
        type=int,
        default=1024,
        help="length of context window in tokens for transformers",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="gpt2",
        help="the path to folder with pretrained models (expected to work with model.from_pretraiend() method)",
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrained",
        help="model label controlling which checkpoint to load",
    )
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--compute_wt103_ppl", action="store_true")

    # ablation params
    parser.add_argument(
        "--ablate_layer_head_dict",
        type=str,
        help="A string formated as dict, specifying which layers and heads to ablate (assuming 0-indexing)."
        "E.g. to ablate heads 0, 1, 2 in layer 1 and heads 5, 6 in layer 5, you type:"
        "'{1:, [0, 1, 2], 5: [5, 6]}'",
    )
    parser.add_argument("--ablate_topk_heads")
    parser.add_argument(
        "--ablate_topk_heads_seed",
        type=int,
        help="Random seed to select random heads for ablation (control experiment)",
    )
    parser.add_argument(
        "--ablate_topk_heads_toi",
        type=str,
        help="List of position indices over which to aggregate attn for top-k selection",
    )
    parser.add_argument(
        "--ablate_params",
        type=str,
        default="Q,K",
        help="Attention parameters to be ablate. Specify as comma-separated string.",
    )
    parser.add_argument(
        "--ablation_type", type=str, choices=["zero", "shuffle"], default="zero"
    )

    # experiment params
    parser.add_argument(
        "--model_seed",
        type=int,
        default=12345,
        help="seed value to be used in torch.manual_seed() prior to calling GPT2Model()",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="whether to run on cpu or cuda",
    )
    parser.add_argument("--model_statedict_filename", type=str, default="")
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
    parser.add_argument(
        "--revision",
        default="main",
        type=str,
        help="str, a revision of a model",
    )
    parser.add_argument(
        "--aggregate_output",
        action="store_true",
        help="whether or not to only output aggregated dataframe",
    )
    parser.add_argument(
        "--aggregate_positions",
        type=str,
        help="string represing a a python list of integers indicating over which positions to aggregate (e.g. '[0, 1]' to aggregate over first two positions)",
    )

    if devtesting:  # manually set this to True if testing code interactively
        input_args = get_input_args_for_testing()

    if input_args:
        argins = parser.parse_args(input_args)
    else:
        argins = parser.parse_args()

    # if test run, populate input arguments manually (these are
    # corresponding to choices in wm_suite.io.test_ds.get_test_data())
    if argins.test_run:
        logger.info("Doing a test run...")
        argins.scenario = "sce1"
        argins.list_len = 3
        argins.prompt_len = "1"

    # ===== INITIATE MODEL AND TOKANIZER CLASS ===== #

    # declare device and paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup the model
    logger.info("Loading tokenizer {}".format(argins.tokenizer))
    tokenizer = AutoTokenizer.from_pretrained(argins.tokenizer)

    # pretrained models
    logger.info("Using {} model".format(argins.model_type))
    logger.info("Loading checkpoint {}".format(argins.checkpoint))
    model = AutoModelForCausalLM.from_pretrained(
        argins.checkpoint, revision=argins.revision
    )

    ismlm = False

    if argins.checkpoint == "gpt2":
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # or initialize a random model
    if argins.model_type == "random":
        # initialize with random weights
        torch.manual_seed(argins.model_seed)
        model = AutoModelForCausalLM.from_config(GPT2Config())

    # permute the weights of gpt-small
    elif argins.model_type == "random-att":
        logger.info("Permuting model attention weights ...\n")
        model = permute_qk_weights(model, per_head=False, seed=argins.model_seed)

    # permute attenion heads of gpt2 small
    elif argins.model_type == "random-att-per-head":
        logger.info("Permuting Q and K weights ...\n")
        model = permute_qk_weights(model, per_head=True, seed=argins.model_seed)

    # shuffle embedding vectors of gpt2 small
    elif argins.model_type == "shuff-wpe":
        logger.info("Permuting token positions in wpe...")
        rng = np.random.RandomState(seed=argins.model_seed)

        wpe = model.transformer.wpe.weight  # shape = (token_positions, embedding_dim)

        # permutation only permutes across 0 dim (rows=token_positions)
        wpe_shuf = torch.tensor(rng.permutation(wpe.detach().numpy()))
        model.transformer.wpe.weight = torch.nn.Parameter(
            data=wpe_shuf, requires_grad=False
        )

    elif "ablated" in argins.model_type:
        logger.info(f"Running ablation experiment")

        if argins.ablate_layer_head_dict is not None:
            logger.info(
                f"Using the following ablation dict:\n{argins.ablate_layer_head_dict}"
            )

            # now set the selected head in selected layer to 0
            lh_dict = literal_eval(argins.ablate_layer_head_dict)

        elif argins.ablate_topk_heads:
            # make sure that these arguments are not passed
            assert argins.ablate_heads is None
            assert argins.ablate_layers is None

            attn_dict = dict(
                np.load(
                    "/scratch/ka2773/project/lm-mem/output/wm_attention/attention_weights_gpt2_colon-colon-p1.npz"
                )
            )

            if argins.ablate_topk_heads_seed:
                seed = argins.ablate_topk_heads_seed
            else:
                seed = 12345  # use a seed just to find_topk_attn runs

            # if specified, find heads that fall both in induciton and
            # matching heads
            if argins.ablate_topk_heads == "induction-matching-intersect":
                lh_dict = find_topk_intersection(
                    attn_dict["data"], tois=([13], [14, 16, 18]), topk=20, seed=12345
                )

            elif argins.ablate_topk_heads == "matching-bottom5":
                # find effect of ablation of onlyt the last 5 matching
                # heads (seems to have disproportionaly strong effect)
                top15_matching, _, _ = find_topk_attn(
                    attn_dict["data"], topk=15, tokens_of_interest=[13], seed=12345
                )
                top20_matching, _, _ = find_topk_attn(
                    attn_dict["data"], topk=20, tokens_of_interest=[13], seed=12345
                )
                top15labels, top20labels = (
                    from_dict_to_labels(top15_matching),
                    from_dict_to_labels(top20_matching),
                )
                resid_labels = list(set(top20labels) - set(top15labels))

                lh_dict = from_labels_to_dict(resid_labels)

            else:
                argins.ablate_topk_heads = int(argins.ablate_topk_heads)
                toi = list(literal_eval(argins.ablate_topk_heads_toi))
                out_tuple = find_topk_attn(
                    attn_dict["data"],
                    argins.ablate_topk_heads,
                    tokens_of_interest=toi,
                    seed=seed,
                )

                lh_dict, lh_dict_ctrl, _ = out_tuple

        # by default use the non-randomly selected heads
        layer_head_dict = lh_dict

        # if seed is provided by the user, use random selection of heads instead
        if argins.ablate_topk_heads_seed:
            logger.info(f"Ablating {argins.ablate_topk_heads} random heads")
            layer_head_dict = lh_dict_ctrl

        # ablate heads by setting GPT2Attention() classes in selected
        # layers to GPT2AttentionAblated()
        model = ablate_attn_module(
            model, layer_head_dict=layer_head_dict, ablation_type="zero"
        )

    # ===== TOKENIZE AND PREPARE THE VIGNETTES ===== #

    if argins.test_run:
        input_sequences = get_test_data()

    else:
        # fname = os.path.join(data_dir, argins.input_filename)

        input_sequences = get_input_sequences(
            condition=argins.condition,
            scenario=argins.scenario,
            list_type="random"
            if argins.noun_list_file == "random_lists.json"
            else "categorized",
            list_len=f"n{argins.list_len}",
            prompt_key=argins.prompt_len,
            tokenizer_name=argins.checkpoint,
            pretokenize_moses=argins.pretokenize_moses,
        )

    for s in input_sequences:
        s.prompt = argins.prompt_len
    # input_sequences_info["prompt"] = [argins.prompt_len for _ in input_sequences_info['list_len']]

    # ===== EXPERIMENT ===== #

    model.eval()  # set to evaluation mode

    # initialize experiment class
    experiment = Experiment(
        model=model,
        ismlm=ismlm,
        tokenizer=tokenizer,
        context_len=argins.context_len,
        batch_size=argins.batch_size,
        stride=1,
        use_cache=False,
        device=device,
    )

    # run experiment
    output_dict = experiment.start(
        input_sequences=[sequence.ids for sequence in input_sequences]
    )

    # ===== WT-103 PERPLEXITY ===== #
    ppl = torch.tensor(torch.nan)

    if argins.compute_wt103_ppl:
        logger.info(f"Loading {PATHS.wt103_test}...")
        _, ids = WikiTextDataset(tokenizer=tokenizer).retokenize_txt(PATHS.wt103_test)

        # initialize experiment class
        experiment2 = Experiment(
            model=model,
            ismlm=ismlm,
            tokenizer=tokenizer,
            context_len=argins.context_len,
            batch_size=argins.batch_size,
            stride=1,
            use_cache=False,
            device=device,
        )

        ppl, _, _ = experiment2.ppl(
            input_ids=torch.tensor([ids]), context_len=1024, stride=256
        )

    # ===== POST PROCESSING ===== #

    logger.info("Postprocessing outputs...")

    # merge tokens and surprisals that were splits into BPE tokens
    output_dict = experiment.merge_bpe_splits(output_dict)

    # find timesteps that correspond to punctuation, end of string and
    # empty slots
    punct_symbols = [
        np.array([s.strip("Ġ") in punctuation for s in l]) for l in output_dict["token"]
    ]  # find positions of punctuation symbols
    eos_symbols = [
        np.array([s.strip("Ġ") in tokenizer.eos_token for s in l])
        for l in output_dict["token"]
    ]  # find positions for end of text symbols
    empty_symbols = [
        np.array([s.strip("Ġ") == "" for s in l]) for l in output_dict["token"]
    ]  # make sure to track empty strings after merging

    # create a joint boolean, per sequence
    timesteps = [
        [(~punct & ~eos & ~empty) for punct, eos, empty in zip(x, y, z)]
        for x, y, z in zip(punct_symbols, eos_symbols, empty_symbols)
    ]

    # create nan arrays that are going to be populated with markers
    # and will have nan values elsewhere
    output_dict["marker_pos"] = [
        np.full(shape=len(s), fill_value=np.nan) for s in output_dict["token"]
    ]
    output_dict["marker_pos_rel"] = [
        np.full(shape=len(s), fill_value=np.nan) for s in output_dict["token"]
    ]

    # now create absolute and relative markers for each sequence
    for i, markers in enumerate([sequence.trial_ids for sequence in input_sequences]):
        sel = timesteps[i]
        output_dict["marker_pos"][i][sel] = code_relative_markers(
            np.array(markers)[sel]
        )[0]
        output_dict["marker_pos_rel"][i][sel] = code_relative_markers(
            np.array(markers)[sel]
        )[1]

    experiment.model.to("cpu")

    # ===== SAVING MODEL AND PERPLEXITY ===== #

    if argins.model_statedict_filename:
        fn = os.path.join(argins.output_dir, argins.model_statedict_filename)
        # logger.info(f"Saving {fn}")
        # torch.save(experiment.model.state_dict(), fn)

    if argins.model_statedict_filename and argins.compute_wt103_ppl:
        # store wikitext-103 perplexity
        ppl_dict = {
            "wt103_ppl": round(ppl.cpu().item(), 2),
            "model": argins.model_statedict_filename,
        }

        fn = os.path.join(
            argins.output_dir,
            argins.model_statedict_filename.replace(".pt", "_ppl.json"),
        )
        with open(fn, "w") as fh:
            json.dump(ppl_dict, fh)

    # ===== FORMAT AND SAVE OUTPUT FILES ===== #

    colnames = ["token", "surp", "marker_pos", "marker_pos_rel"]
    metacols = [
        "ppl",
        "scenario",
        "second_list",
        "list",
        "trialID",
        "positionID",
        "subtok",
        "list_len",
        "prompt_len",
    ]

    scenarioid2label = {
        "sce1": "intact",
    }

    arrays = [
        output_dict["token"],
        output_dict["surp"],
        output_dict["marker_pos"],
        output_dict["marker_pos_rel"],
    ]

    metarrays = [
        output_dict["sequence_ppl"],
        [
            scenarioid2label[argins.scenario] for _ in range(len(input_sequences))
        ],  # "sce1" | "sce2" | etc.
        [
            argins.condition for _ in range(len(input_sequences))
        ],  # "repeat" | "permute" | "control"
        [
            argins.list_type for _ in range(len(input_sequences))
        ],  # "random" | "categorized"
        [sequence.trial_ids for sequence in input_sequences],
        [sequence.position_ids for sequence in input_sequences],
        [sequence.subtok_ids for sequence in input_sequences],
        [sequence.list_len for sequence in input_sequences],
        [sequence.prompt for sequence in input_sequences],
    ]

    # put everything into a dataframe
    logger.info("Converting to dataframe...")

    output_df = outputs2dataframe(colnames, arrays, metacols, metarrays)

    for col in ("prompt_len", "list_len"):
        output_df[col] = output_df[col].astype(int)

    output_df["model"] = argins.model_type
    output_df = output_df.rename(
        columns={"trialID": "marker", "stimID": "stimid", "scenario": "context"}
    )
    output_df["model_id"] = argins.model_id

    output_df.attrs = {"wt103_ppl": np.round(ppl.cpu().item(), 2), "argins": argins}

    # drop nan rows (merged BPE timesteps)
    output_df = output_df.loc[~output_df.token.str.strip("Ġ").isin([""]), :]

    # this is the variable that is returned
    output = output_df

    # ===== COMPUTE REPEAT SURPRISAL IF NEEDED ===== #
    if argins.aggregate_output:
        variables = [
            {"list_len": [int(argins.list_len)]},
            {"prompt_len": [int(argins.prompt_len)]},
            {"context": ["intact"]},
            {
                "marker_pos_rel": literal_eval(argins.aggregate_positions)
            },  # average over first timestep
        ]

        # create dict for storing output measures
        outdict = {
            "x1": {
                "median": None,
                "mad": None,
                "minmax": None,
                "ci95": None,
            },  # raw surprisal on first list
            "x2": {
                "median": None,
                "mad": None,
                "minmax": None,
                "ci95": None,
            },  # raw surprisal on second list
            "rs": {
                "median": None,
                "mad": None,
                "minmax": None,
                "ci95": None,
            },  # reapeat surprisal
            "model_id": None,
        }

        # compute repeat surprisal
        output_agg, _ = filter_and_aggregate(
            output_df,
            model=argins.model_type,
            model_id=output_df.model_id.unique().item(),
            groups=variables,
            aggregating_metric="mean",
        )

        x1 = output_agg.x1.to_numpy()
        x2 = output_agg.x2.to_numpy()
        y = output_agg.x_perc.to_numpy()

        # repeat surprisal
        outdict["rs"]["median"] = np.median(y)
        outdict["rs"]["minmax"] = (np.min(y), np.max(y))
        outdict["rs"]["mad"] = median_abs_deviation(y)
        ci = bootstrap(
            (y,),
            axis=0,
            statistic=np.median,
            confidence_level=0.95,
            n_resamples=10000,
            random_state=np.random.RandomState(12345),
        )
        outdict["rs"]["ci95"] = (
            ci.confidence_interval.low,
            ci.confidence_interval.high,
        )

        # first list
        outdict["x1"]["median"] = np.median(x1)
        outdict["x1"]["minmax"] = (np.min(x1), np.max(x1))
        outdict["x1"]["mad"] = median_abs_deviation(x1)
        ci = bootstrap(
            (x1,),
            axis=0,
            statistic=np.median,
            confidence_level=0.95,
            n_resamples=10000,
            random_state=np.random.RandomState(12345),
        )
        outdict["x1"]["ci95"] = (
            ci.confidence_interval.low,
            ci.confidence_interval.high,
        )

        # second list
        outdict["x2"]["median"] = np.median(x2)
        outdict["x2"]["minmax"] = (np.min(x2), np.max(x2))
        outdict["x2"]["mad"] = median_abs_deviation(x2)
        ci = bootstrap(
            (x2,),
            axis=0,
            statistic=np.median,
            confidence_level=0.95,
            n_resamples=10000,
            random_state=np.random.RandomState(12345),
        )
        outdict["x2"]["ci95"] = (
            ci.confidence_interval.low,
            ci.confidence_interval.high,
        )

        outdict["model_id"] = argins.model_id

        outdict["wt103_ppl"] = ppl.cpu().item()

        output = outdict

    # save output
    if argins.output_dir and argins.output_filename:
        outpath = os.path.join(argins.output_dir, argins.output_filename)
        logger.info("Saving {}".format(os.path.join(outpath)))

        # save the full, non-aggregated dataframe
        if isinstance(output, pd.DataFrame):
            output.to_csv(outpath, sep="\t")

        # or save the aggregatec metrics in json
        elif isinstance(output, dict):
            with open(outpath, "w") as fh:
                json.dump(output, fh, indent=4)

    return output


# ===== RUN ===== #
if __name__ == "__main__":
    main()
