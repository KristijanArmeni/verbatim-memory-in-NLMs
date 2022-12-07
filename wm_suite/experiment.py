# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:47:27 2021

@author: karmeni1
 """

import os
import numpy as np
import logging
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from nltk import word_tokenize, sent_tokenize
from types import SimpleNamespace
import sys

# own modules
sys.path.append("/home/ka2773/project/lm-mem/src/src/wm_suite/")
sys.path.append("/home/ka2773/project/lm-mem/src/src/wm_suite/awd_lstm")
from models.rnn.model import RNNModel
from models.awd_lstm.model import RNNModel as AWD_RNNModel
from models.awd_lstm.utils import repackage_hidden, batchify
from models.awd_lstm.splitcross import SplitCrossEntropyLoss


logging.basicConfig(format="[INFO: %(funcName)20s()] %(message)s", level=logging.INFO)

# ===== MAIN EXPERIMENT CLASS ===== #


def isfloat(instr):
    """ Reports whether a string is floatable """
    try:
        _ = float(instr)
        return(True)
    except:
        return(False)


class Dictionary(object):
    """ Maps between observations and indices

    KA: from github.com/vansky/neural-complexity-master
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """ Adds a new obs to the dictionary if needed """

        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Dataset(object):

    def __init__(self, metadict):

        self.seqs = []          # input sequences
        self.seq_ids = []

        metainfo = {'sentid': []}

        # add any additional stimulus information if provided
        if len(metadict) > 0:
            metainfo.update(metadict)

        self.meta = SimpleNamespace(**metainfo)
        self.dictionary = Dictionary()


    def load_dict(self, path):
        """ Loads dictionary from disk """

        assert os.path.exists(path), "Bad path: %s" % path
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'rb') as file_handle:
                fdata = torch.load(file_handle, pickle_module=dill)
                if isinstance(fdata, tuple):
                    # Compatibility with old pytorch LM saving
                    self.dictionary = fdata[3]
                self.dictionary = fdata
        else:
            # Assume dict is plaintext
            with open(path, 'r', encoding='utf-8') as file_handle:
                for line in file_handle:
                    self.dictionary.add_word(line.strip())

        logging.info("Added Dictionary() class to .dictionary attribute...")

    def tokenize_input_sequences(self, input_sequences, tokenizer_func=None):

        logging.info('Tokenizing input sequences ...')

        for seq in tqdm(input_sequences, desc="sequence"):

            toks = tokenizer_func(seq)
            self.seqs.append(["<eos>"] + toks + ["<eos>"])

            # add sentence information
            sent_strings = sent_tokenize(" ".join(toks))
            tmp = []
            for j, sent in enumerate(sent_strings):
                ids = np.zeros(shape=len(sent.split(" ")), dtype=np.int64) + j
                tmp.append(ids)

            self.meta.sentid.append(np.concatenate(tmp))


    def tokens_to_ids(self, check_unkns=True):

        logging.info("Converting tokens to ids ...")

        for seq in self.seqs:

            ids = []

            for i, word in enumerate(tqdm(seq, desc="sequence: ")):

                # code OOV as <unk> and numerals to <num>
                if word not in self.dictionary.word2idx:

                    ids.append(self.dictionary.add_word("<unk>"))

                elif isfloat(word) and '<num>' in self.dictionary.word2idx:

                    ids.append(self.dictionary.add_word("<num>"))

                else:
                    ids.append(self.dictionary.word2idx[word])

            # convert to tensor, add batch and feature dimension
            self.seq_ids.append(torch.tensor(ids,  dtype=torch.int64)
                                .unsqueeze(-1)
                                .unsqueeze(-1))


def read_marker_file(path):

    # read markers.txt line by line and store the contents to markers
    with open(path, 'r') as file_handle:

        # read variable names from header (first row)
        # we are currently assuming at least three columns below
        colnames = file_handle.readline().strip("\n").split("\t")
        markers = {key: [] for key in colnames}

        for line in file_handle:
            # read the line containing marker values and prompt labels
            # and make it a list (e.g [0, 0, 0, 0, 1, 1, 1, 1])
            # first (markers, condition_label1, condition_label2)
            row_values = line.strip("\n").split("\t")

            tmp = [int(el) for el in row_values[0].strip("[]").split(",")]
            markers[colnames[0]].append(tmp)            # these are the actual markers
            markers[colnames[1]].append(row_values[1])  # this is stimid
            markers[colnames[2]].append(row_values[2])  # this codes ex. condition
            markers[colnames[3]].append(row_values[3])  # this codes ex. condition

    return markers

class ModelOutput(object):

    def __init__(self):

        self.labels = []
        self.trial = None
        self.meta = None
        self.step = []
        self.tokenid = []
        self.log_probs = None
        self.ppl = []
        self.states = None

    def __repr__(self):

        return "ModelOutput(s{})".format(self.trial+1)

    def info(self):

        prefix = "ModelOutput() class with attributes:\n\n"
        info = prefix + \
               "self.labels\n" + \
               "self.trial\n" + \
               "self.meta\n" + \
               "self.step\n" + \
               "self.tokenid\n" + \
               "self.log_probs\n" + \
               "self.states\n"

        print(info)

    def to_numpy_arrays(self):

        self.labels = np.asarray(self.labels)
        self.log_probs = np.asarray(self.log_probs)
        self.step = np.asarray(self.step)

        return self

    def drop_eos_tokens(self):

        sel = np.where(self.labels == '<eos>')[0]

        self.labels = np.delete(self.labels, sel, axis=0)
        self.log_probs = np.delete(self.log_probs, sel, axis=0)
        self.step = np.delete(self.step, sel, axis=0)

        return self

    def drop_first_last(self):
        """
        wrapper to drop first and last time steps wich are <eos>
        """
        self.labels = self.labels[1:-1]
        self.log_probs = self.log_probs[1:-1]
        self.step = self.step[1:-1]

        return self

class Experiment(object):

    def __init__(self, model, rnn_type, criterion, store_states, dictionary, device):

        self.model = model
        self.rnn_type = model.rnn_type if rnn_type is None else rnn_type
        self.criterion = criterion if criterion is not None else None
        self.store_states = store_states
        self.dictionary = dictionary
        self.device = device

    def run(self, input_set, metainfo):
        """.run(input_set) is the highest-level loop that loops over
        the entire stimulus set (sentences, stories etc.)
        """

        for key in metainfo.keys():
            assert len(input_set) == len(metainfo[key])

        logging.info("Extracing model outputs ...")

        output = []

        for j, sequence in enumerate(tqdm(input_set, desc="sequence")):

            if self.store_states:
                self.model.rnn.init_state_logger()
                states = self.model.rnn.states
                store_states = self.model.rnn.store_states

            else:  # initiate empty lists otherwise
                states = [[] for l in range(self.model.nlayers)]
                store_states = False

            if self.rnn_type in ["QRNN", "AWD_LSTM"]:
                eval_bs = 1
                sequence = batchify(sequence, bsz=eval_bs, cuda=True if self.device.type == 'cuda' else False)
                model_output = self.evaluate_awd(data_source=sequence, 
                                                  hidden_size=self.model.ninp,
                                                  bptt=1,
                                                  batch_size=eval_bs)
            else:
                model_output = self.get_states(input_sequence=sequence.to(self.device))

             # log trial id and append to experiment output
            model_output.trial = j

            model_output.states = [s.to_numpy_arrays()
                                   if store_states else s
                                   for s in states]

            model_output.meta = {key: metainfo[key][j] for key in metainfo.keys()}

            # conver .step and .log_probs to arrays
            output.append(model_output
                          .to_numpy_arrays()
                          .drop_eos_tokens())

        return output

    @staticmethod
    def perplexity(neg_log_likelihoods):

        """
        Parameters:
        ----------
        neg_log_likelihoods : list
            list of elements which represent neg. log. likelihood per token

        Returns:
        --------
        ppl : scalar
            np.exp(np.nanmean(neg_log_likelihoods))
        """

        return np.exp(np.nanmean(neg_log_likelihoods))

    def get_states(self, input_sequence):
        """.run_trial implements a loop over a single continuous sequence
        (sentence, paragraph etc.). It initializes hiddens states as zero
        vectors.

        """
        # initialize hidden state and cell state (h_0, c_0)
        hidden = self.model.init_hidden(bsz=1)

        # initialize the output class
        output = ModelOutput()

        # initialize loss, CrosEntropyLoss() expects logits as inputs
        # it applies the nn.LogSoftmax() and nn.LLLoss() internally
        # https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
        loss = CrossEntropyLoss()

        # create targets tensor, pad the first token position with -100
        targets = input_sequence.clone()

        output.log_probs = np.zeros(shape=targets.shape)

        predictions = None

        # loop over this sequence
        for i, x in enumerate(input_sequence):

            if predictions is None:
                # on step 0, there is no loss, make it nan
                output.log_probs[i] = np.nan
            else:
                # compute loss for logits i-1 and target i ('targets' are thus shifted)
                # predictions.shape = (batch, seq_len, classes)
                # targets.shape = (seq_len, batch, N)

                nll = loss(predictions[:, 0, :], target=targets[i, :, 0])
                output.log_probs[i] = nll.detach().item()

            # get logits and hidden from the forward pass
            predictions, next_hidden = self.model(x, hidden)

            # store the token strings for each corresponding time step
            output.labels.append(self.dictionary.idx2word[x.item()])

            # update hidden variable
            hidden = next_hidden

            # track step index
            output.step.append(i)

        # compute ppl for input sequence
        output.ppl.append(self.perplexity(output.log_probs))

        return output

    def evaluate_awd(self, data_source, hidden_size, bptt, batch_size=10):
        """
        A manual copy of main.evaluate from the salesforce repo
        """
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        if self.rnn_type == "QRNN":
            self.model.reset()

        total_loss = 0

        out = ModelOutput()
        out.log_probs = []

        hidden = self.model.init_hidden(batch_size)
        
        for i in range(0, data_source.size(0) - 1, bptt):

            data, targets = get_batch_awd(data_source, i, bptt, seq_len=None, evaluation=True)

            output, hidden = self.model(data, hidden)

            loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets).data
            
            #if need be for testing
            #print(f"Input: {self.dictionary.idx2word[data.item()]} | target: {self.dictionary.idx2word[targets.item()]} | loss: {loss.detach().cpu().item()}")

            out.log_probs.append(loss.detach().cpu().item())

            out.labels.append(self.dictionary.idx2word[targets.item()])
            out.step.append(i)

            total_loss += len(data) * loss
        
            hidden = repackage_hidden(hidden)
        
        return out

def get_batch_awd(source, i, bptt, seq_len=None, evaluation=False):

    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate_awd_standalone(model, data_source, criterion, bptt, batch_size=10):
    # Turn on evaluation mode which disables dropout.

    total_loss = 0

    hidden = model.init_hidden(batch_size)

    for i in trange(0, data_source.size(0) - 1, bptt):

        data, targets = get_batch_awd(data_source, i, bptt, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)

    return total_loss.item() / len(data_source)

def get_args_for_dev(checkpoint_folder, 
                     model_weights, 
                     vocab_file, 
                     config_file, 
                     input_file,
                     marker_file,
                     per_token_output,
                     output_folder,
                     output_filename):

    args = {

        "checkpoint_folder": checkpoint_folder,
        "model_weights": model_weights,
        "vocab_file": vocab_file,
        "config_file": config_file,
        "input_file": input_file,
        "marker_file": marker_file,
        "per_token_output": per_token_output,
        "output_folder": output_folder,
        "output_filename": output_filename
    }

    return args

# ===== RUNTIME CODE ===== #
if __name__ == "__main__":

    import json
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_folder", type=str,
        help="path/to/folder in which model is save")
    parser.add_argument("--model_weights", type=str,
        help="path/to/model_weights.pt of LSTM model")
    parser.add_argument("--vocab_file", type=str, default="./rnn_/vocab.text",
        help="path/to/vocab_file.txt which contains all tokens in model's vocabulary")
    parser.add_argument("--config_file", type=str,
        help="path/to/config_file.json which contains the LSTM model's configuration")
    parser.add_argument("-i", "--input_file", type=str,
        help="path/to/input_file.txt which contains input sentences")
    parser.add_argument("--marker_file", type=str, default="",
        help="(optional) empty string or path/to/marker_file.txt which contains metadata information about"\
                + " each input sequence (default: empty string)")
    parser.add_argument("--per_token_output", action="store_true")
    parser.add_argument("--output_folder", type=str,
        help="path/to/output_folder in which all outputs will be saved")
    parser.add_argument("--output_filename", type=str,
        help="output_file.csv - file with results")

    args = parser.parse_args()

    #args = SimpleNamespace(**get_args_for_dev(checkpoint_folder="/home/ka2773/project/lm-mem/src/src/wm_suite/awd_lstm",
    #                        model_weights="/scratch/ka2773/project/lm-mem/checkpoints/awd_lstm/LSTM_3-layer_adam.pt",
    #                        vocab_file="awd_lstm_corpus",
    #                        config_file="/home/ka2773/project/lm-mem/src/src/wm_suite/awd_lstm/AWD-LSTM_3-layer_config.json",
    #                        input_file="/home/ka2773/project/lm-mem/src/data/rnn_input_files/categorized_lists_sce1_control.txt",
    #                        marker_file="/home/ka2773/project/lm-mem/src/data/rnn_input_files/categorized_lists_sce1_control_markers.txt",
    #                        per_token_output=True,
    #                        output_folder="/scratch/ka2773/project/lm-mem/output/awd_lstm",
    #                        output_filename="awd-lstm_test.csv"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device = {}".format(device))

    # load json file, should be a list of strings where eash string
    # (can be multiple sentences) is a single input stream to the model
    with open(args.input_file, "r") as f:
        
        logging.info("Loading {}".format(args.input_file))
        
        # load .json otherwise assume you can read input sequences line by line
        if ".json" in args.input_file:
            input_set = json.load(f)
        else:
            input_set = [l.strip("\n").lower() for l in f.readlines()]

    # read in the marker files where (optional) metadata about input sequences is stored
    if args.marker_file:
        markers = read_marker_file(args.marker_file)
    # otherwise use empty dict, Dataset() can deal with this
    else:
        markers = {}

    # initialize dataset class
    if args.vocab_file == "awd_lstm_corpus":
        data = torch.load("/home/ka2773/project/lm-mem/src/src/wm_suite/awd_lstm/data/wikitext-103/corpus.844da84f12f45a4f741beb331daea986.data")
        ds = Dataset(metadict=markers)
        ds.dictionary = data.dictionary
    else:
        ds = Dataset(metadict=markers)
        ds.load_dict(path=args.vocab_file)

    # this is independent from the dictionary used
    ds.tokenize_input_sequences(input_sequences=input_set, tokenizer_func=word_tokenize)
    # convert tokenized strings to torch.LongTensor, this will write to
    # .seq and .seq_ids attributes
    ds.tokens_to_ids()

    # read in the configuration file which contains values for model parameters
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # instantiate the loss
    splits = [2800, 20000, 76000]  # these are hard-coded
    criterion = SplitCrossEntropyLoss(config["n_hid"], splits, verbose=False).to(device)

    # initialize model class and use the appropriate RNNModel class
    if config["name"] == "LSTM":

        model = RNNModel(rnn_type='LSTM',
                        ntoken=config['n_vocab'],
                        ninp=config['n_inp'],
                        nhid=config['n_hid'],
                        nlayers=config['n_layers'],
                        store_states=False)

        
        # make this input argument at some point
        print(device)
        model.load_state_dict(
            torch.load(args.model_weights, map_location=torch.device(device)))

    # if running AWD_LSTM, make sure we're running pytorch 0.4 :/
    elif config["name"] == "AWD_LSTM":

        model = AWD_RNNModel(rnn_type="LSTM",
                              ntoken=config['n_vocab'],
                              ninp=config['n_inp'],
                              nhid=config['n_hid'],
                              nlayers=config['n_layers'],
                              dropout=config['dropout'],
                              dropoute=config['dropoute'],
                              dropouth=config['dropouth'],
                              dropouti=config['dropouti'],
                              wdrop=config['wdrop'],
                              tie_weights=config['tie_weights'],
                            )

        model.rnn = None  #TEMP, add this dummy attribute to make code below run

        model, criterion, _ = torch.load(args.model_weights)
        
    
    # this is a QRNN-specific method
    if config["name"] == "QRNN":
        model.reset()
    
    model.eval()

    print(model)

    # use a copy of evaluate from awd_lstm.main.py to get logprobs for QRNN/AWD_LSTM
    eval_batch_size=1

    # ===== EXPERIMENT CLASS ===== #
    logging.info("Running experiment...")
    logging.info("Per token output == {}".format(args.per_token_output))

    exp = Experiment(model=model.to(device), criterion=criterion, rnn_type="AWD_LSTM", store_states=False, 
                     dictionary=ds.dictionary, device=device)

    outputs = exp.run(input_set=ds.seq_ids, metainfo=ds.meta.__dict__)

    # convert the list of ModelOuput() classes into a list of dicts
    # better serialize with built-in types
    datadict = [{'trial': o.trial,
                 'token': o.labels,
                 'step': o.step,
                 'stimid': o.meta['stimid'] if 'stimid' in o.meta.keys() else np.nan,
                 'markers': o.meta['markers'] if 'markers' in o.meta.keys() else np.nan,
                 'list_len': o.meta['list_len'] if 'list_len' in o.meta.keys() else np.nan,
                 'prompt_len': o.meta['prompt_len'] if 'prompt_len' in o.meta.keys() else np.nan,
                 'ppl': o.ppl,
                 'log_probs': np.squeeze(o.log_probs) if o.log_probs.ndim > 1 else o.log_probs,
                 'states': [SimpleNamespace(**vars(s)) if s else [] for s in o.states],}
                  for o in outputs]

    # convert dict to a SimpleNamespace to have dot indexing instead of brackets
    data = [SimpleNamespace(**el) for el in datadict]

    # wrapper
    def datanamespace_to_df(data):

        dfs = []

        for i, el in enumerate(tqdm(data, desc="sequence")):
            
            # if per token output is to be saved, create appropriate columns in data frame
            if args.per_token_output:
                cols = ["word", "corpuspos", "markers", "surp"]

                input_arrays = [np.array(el.token),
                                el.step,
                                el.markers,
                                el.log_probs]

                tmp = pd.DataFrame(data=np.array(input_arrays).T, columns=cols)
                
                tmp['stimid'] = el.stimid
                tmp['list_len'] = el.list_len     # read-in from markers file
                tmp['prompt_len'] = el.prompt_len # read-in from markers file
                tmp['stimid'] = el.stimid         # read-in from markers file, counts over conditions (0-229)

            else:
                # only store perplexity if per token loss in not stored
                tmp = pd.DataFrame()
                tmp['ppl'] = el.ppl

            tmp['sentid'] = el.trial     # this counts over all input sequences in the experiment

            dfs.append(tmp)

        return pd.concat(dfs)
    
    # convert data struct to data frame
    df = datanamespace_to_df(data)
    
    print(df.markers.unique())

    # save output as csv
    savename = os.path.join(args.output_folder, args.output_filename)

    logging.info('Saving {}'.format(savename))
    df.to_csv(savename, sep="\t")

