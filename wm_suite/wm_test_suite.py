"""
surprisal.py is used to run the perplexity experiment with GPT-2
it relies on the Experiment() class which is just a class with wrapper methods
around the Transformers library.

Use as:

python experiment.py
or from an ipython console:
%run experiment.py ""

"""

import json
import sys, os
import argparse
import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, \
                         BertForMaskedLM, TransfoXLLMHeadModel,  TransfoXLTokenizer, \
                         AutoTokenizer, top_k_top_p_filtering
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import trange
from typing import List, Dict, Tuple, Any
from string import punctuation
from tqdm import tqdm
import logging

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

    def __init__(self, model, ismlm, tokenizer, context_len, batch_size, use_cache, device):

        self.model = model
        self.ismlm = ismlm
        self.device = device
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.batch_size = batch_size
        self.use_cache = use_cache

        self.loss_fct_batched = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.to(device)
        

    def get_logits(self, input_string):

        inp_ids = self.tokenizer.encode(input_string, return_tensors="pt")

        # take the output logits of the model
        with torch.no_grad():
            logits = self.model(inp_ids)[0]  # tuple of shape = (logits, layer and attention output)

        return logits

    def get_probs(self, logits, n_tokens=5, k=10^3, p=0.9):

        """get_probs(self, logtis, k, p) is a convenience fun wrapping around
        top_k_top_p_filtering(logits, top_k, top_p), calling F.softmax() on the output and sampling from   there
        """

        # take only the top 10 logits
        filtered_logits = top_k_top_p_filtering(logits[:, -1, :], top_k=k, top_p=p)

        # apply the softmax to the truncated distribution
        probs = F.softmax(filtered_logits, dim=-1)

        # now sample n tokens
        # next_tokens = torch.multinomial(probs, num_samples=n_tokens, replacement=False)

        token_prob, token_ids = torch.topk(x=probs, dim=1, k=n_tokens, sorted=True)

        return token_prob, token_ids

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

            if i in range(0, 5):
                print(f"inputs: {sel_input_ids}")
                print(f"targets: {target_ids}")

            # set model to evaluation mode
            self.model.eval()

            # get model output
            with torch.no_grad():

               # compute neg log likelihood over target ids (n+1 in our case)
               # indices are shifted under the hood by model.__call__()
               outputs = self.model(sel_input_ids, labels=target_ids)
               
               log_likelihood = outputs.loss.item() * trg_len  # not sure about this multiplication here (undoing averaging?)

               llh.append(log_likelihood)
               toks = self.tokenizer.decode(target_ids[0][-stride::])
               tokens.append(toks)  # store the last token (target_id)

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / end_loc).cpu()
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
                losses = self.loss_fct_batched(
                    outputs.logits[:, -1, :].view(-1, outputs.logits.size(-1)), target_ids[:, -1].view(-1)
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

    def start_sequential(self, input_sequences_ids=None) -> List:
        """
        experiment.start() will loop over prefixes, prompts, and word_lists and run the .ppl() method on them
        
        Parameters:
        ---------
        
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


    def start_batched(self, input_sequences: List[torch.Tensor]):

                # 1. collate_fn
        def get_batch(sequence_list: List[torch.Tensor]) -> Tuple[torch.Tensor]:
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

        # 2. make dataset and dataloader
        sequence_dataset = SimpleDataset(input_sequences)

        sequence_loader = DataLoader(sequence_dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=False, 
                                     collate_fn=get_batch,
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
                                                stride=1,
                                                seq_lens=sequence_lengths, 
                                                targets=targets)

            # store the outputs and
            outputs["sequence_ppl"].extend(ppls)
            outputs["surp"].extend(surprisals)

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


# ===== helper function for code development ===== #
def get_argins_for_dev(setup=False, 
                       inputs_file=None, 
                       inputs_file_unmasked=None,
                       inputs_file_info=None,
                       context_len=1024, 
                       checkpoint="gpt2",
                       tokenizer=None,
                       model_type=None,
                       model_seed=12345,
                       device="cuda",
                       ):

    argins = {

        "setup": setup,
        "inputs_file": inputs_file,
        "inputs_file_info": inputs_file_info,
        "inputs_file_unmasked": inputs_file_unmasked,
        "context_len": context_len,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "model_type": model_type,
        "model_seed": model_seed,
        "device": device

    }

    return argins

# ===== RUNTIME CODE WRAPPER ===== #
def runtime_code():

    # ===== INITIATIONS ===== #
    from types import SimpleNamespace

    # collect input arguments
    parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")

    parser.add_argument("--setup", action="store_true",
                        help="downloads and places nltk model and Tokenizer")
    parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
                        help="str, which scenario to use")
    parser.add_argument("--condition", type=str)
    parser.add_argument("--inputs_file", type=str, help="json file with input sequence IDs which are converted to tensors")
    parser.add_argument("--inputs_file_info", type=str, help="json file with information about input sequences")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--context_len", type=int, default=1024,
                        help="length of context window in tokens for transformers")
    parser.add_argument("--model_type", type=str,
                        help="model label controlling which checkpoint to load")
    # To download a different model look at https://huggingface.co/models?filter=gpt2
    parser.add_argument("--checkpoint", type=str, default="gpt2",
                        help="the path to folder with pretrained models (expected to work with model.from_pretraiend() method)")
    parser.add_argument("--model_seed", type=int, default=12345,
                        help="seed value to be used in torch.manual_seed() prior to calling GPT2Model()")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        help="whether to run on cpu or cuda")
    parser.add_argument("--output_dir", type=str,
                        help="str, the name of folder to write the output_filename in")
    parser.add_argument("--output_filename", type=str,
                        help="str, the name of the output file saving the dataframe")

    argins = parser.parse_args()

    #argins = SimpleNamespace(**get_argins_for_dev(inputs_file="/home/ka2773/project/lm-mem/src/data/transformer_input_files/bert-base-uncased_repeat_sce1_4_n5_random.json",
    #                                              inputs_file_info="/home/ka2773/project/lm-mem/src/data/transformer_input_files/bert-base-uncased_repeat_sce1_4_n5_random_info.json",
    #                                              checkpoint="bert-base-uncased",
    #                                              tokenizer="bert-base-uncased"))

    if argins.setup:
        setup()

    sys.path.append("/home/ka2773/project/lm-mem/src/data/")
    
    # ===== LOAD INPUTS ===== #
    with open(argins.inputs_file, "r") as fh:
        input_sequences = [torch.tensor(e) for e in json.load(fh)]

    with open(argins.inputs_file_info, "r") as fh:
        input_sequences_info = json.load(fh)

    # ===== INITIATE EXPERIMENT CLASS ===== #

    # declare device and paths
    device = torch.device(argins.device if torch.cuda.is_available() else "cpu")

    # setup the model
    logging.info("Loading tokenizer {}".format(argins.path_to_tokenizer))
    tokenizer = GPT2TokenizerFast.from_pretrained(argins.path_to_tokenizer)

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

    # set to evaluation mode
    model.eval()

    #initialize experiment class
    experiment = Experiment(model=model, ismlm=ismlm,
                            tokenizer=tokenizer,
                            context_len=argins.context_len,
                            batch_size=10,
                            use_cache=False,
                            device=device)

    # list storing output dataframes
    experiment_outputs = []

    # ===== RUN EXPERIMENT LOOP ===== #

    output_dict = experiment.start(input_sequences = input_sequences)

    # add token information to the output dictionary
    output_dict["token"] = [experiment.tokenizer.convert_ids_to_tokens(e[0]) for e in input_sequences]

    # ===== FORMAT AND SAVE OUTPUT ===== #

    meta_cols = ["trialID", "positionID", "subtok"]
    colnames = ["token"] + meta_cols + ["surp"]
    arrays = [output_dict["token"],
                input_sequences_info["trialID"],
                input_sequences_info["positionID"],
                input_sequences_info["subtok"],
                output_dict["surp"]]


    counter = 1  # counter for trials
    n_sequences = len(output_dict["surp"])

    # make a new single dict with all rows for the df below
    dfrows = {key_arr[0]: key_arr[1] for i, key_arr in enumerate(zip(colnames, arrays))}

    # loop over trials
    for i in trange(0, n_sequences):

        # a dict of lists (row values) for this sequence
        row_values = {key: np.array(dfrows[key][i]) for key in dfrows.keys()}

        # convert the last two elements of the tuple to an array
        dftmp = pd.DataFrame(row_values)

        # now add the constant values for the current sequence rows
        dftmp["stimid"] = input_sequences_info["stimid"][i]                # stimulus id
        dftmp['prompt'] = input_sequences_info["prompt"][i]                # add a column of prompt labels
        dftmp["list_len"] = input_sequences_info["list_len"][i]            # add list length
        dftmp['stimID'] = counter                               # track sequence id
        dftmp['second_list'] = argins.condition                 # log condition of the second list

        experiment_outputs.append(dftmp)
        counter += 1

    experiment.model.to("cpu")

    # put into a single df and save
    output = pd.concat(experiment_outputs)

    # save output
    outpath = os.path.join(argins.output_dir, argins.output_filename)
    print("Saving {}".format(os.path.join(outpath)))
    output.to_csv(outpath, sep=",")

# ===== RUN ===== #
if __name__ == "__main__":

    runtime_code()
