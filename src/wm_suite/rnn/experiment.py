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
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from types import SimpleNamespace
import pickle

# plotting
from matplotlib import pyplot as plt
plt.ion()

import matplotlib.colors as colors

# own modules
from model import RNNModel

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

        self.step = np.asarray(self.step)

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

    def __init__(self, model, dictionary, device):

        self.model = model
        self.model_with_states = hasattr(model.rnn, 'states')
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

        for j, sequence in enumerate(input_set):

            if self.model_with_states and self.model.rnn.store_states:
                self.model.rnn.init_state_logger()
                states = self.model.rnn.states
                store_states = self.model.rnn.store_states

            else:  # initiate empty lists otherwise
                states = [[] for l in range(self.model.rnn.num_layers)]
                store_states = False

            model_output = self.get_states(input_sequence=sequence.to(self.device))

             # log trial id and append to experiment output
            model_output.trial = j


            model_output.states = [s.to_numpy_arrays()
                                   if store_states else s
                                   for s in states]

            model_output.meta = {key: metainfo[key][j] for key in metainfo.keys()}

            output.append(model_output
                          .to_numpy_arrays()
                          .drop_first_last())

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
        h_0, c_0 = self.model.init_hidden(bsz=1)

        # .h_past and .c_past store states from the past step
        hidden = (h_0, c_0)

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
        for i, x in enumerate(tqdm(input_sequence, desc="sequence: ")):

            if predictions is None:
                # on step 0, there is no loss, make it nan
                output.log_probs[i] = np.nan
            else:
                # compute loss for logits i and target i+1 ('targets' are shifted)
                # predictions.shape = (batch, seq_len, classes)
                # targets.shape = (seq_len, batch, N)
                nll = loss(predictions[:, 0, :], target=targets[i, :, 0])
                output.log_probs[i] = nll.detach().item()

            # get logits and hidden from the forward pass
            predictions, next_hidden = self.model(observation=x, hidden=hidden)

            # store the token strings for each corresponding time step
            output.labels.append(self.dictionary.idx2word[x.item()])

            # update hidden variable
            hidden = next_hidden

            # track step index
            output.step.append(i)

        # compute ppl for input sequence
        output.ppl.append(self.perplexity(output.log_probs))

        return output


def plot_trial(data, batch=0):

    fig1 = plt.figure(constrained_layout=True,
                     num="Loss and hidden (trial {})".format(data.trial))

    widths = [8]
    heights = [4, 6, 6, 6, 6]
    grid = fig1.add_gridspec(ncols=1, nrows=5, width_ratios=widths,
                            height_ratios=heights)
    axes = []
    for row in range(len(heights)):
        for col in range(len(widths)):
            axes.append(fig1.add_subplot(grid[row, col]))

    axes[0].plot(data.step, data.log_probs, '-o')
    axes[0].set_title("surprisal (ppl = {})".format(np.round(data.ppl[0], 2)))
    axes[0].set_xticks(data.step)
    axes[0].set_ylabel('neg LL')

    im = axes[1].imshow(data.h1[:, batch, :].T,
                        cmap = "coolwarm",
                        norm=colors.CenteredNorm(),
                        aspect="auto")
    axes[1].set_title('hidden (L1)')
    axes[1].set_xticks(data.step)
    axes[1].set_ylabel('unit idx')
    fig1.colorbar(im, ax=axes[1])

    im = axes[2].imshow(data.c1[:, batch, :].T,
                        cmap = "coolwarm",
                        norm=colors.CenteredNorm(),
                        aspect="auto")
    axes[2].set_title('cell state (L1)')
    axes[2].set_xticks(data.step)
    axes[2].set_ylabel('unit idx')
    fig1.colorbar(im, ax=axes[2])

    im = axes[3].imshow(data.h2[:, batch, :].T,
                        cmap="coolwarm",
                        norm=colors.CenteredNorm(),
                        aspect="auto")
    axes[3].set_title('hidden (L2)')
    axes[3].set_xticks(data.step)
    axes[3].set_ylabel('unit idx')
    fig1.colorbar(im, ax=axes[3])

    im = axes[4].imshow(data.c2[:, batch, :].T,
                        cmap = "coolwarm",
                        norm=colors.CenteredNorm(),
                        aspect="auto")
    axes[4].set_title('cell state (L2)')
    axes[4].set_xticks(data.step)
    axes[4].set_ylabel('unit idx')
    axes[4].set_xticks(data.step)
    axes[4].set_xticklabels(data.token, rotation=35)
    fig1.colorbar(im, ax=axes[4])


    # ===== LAYER 1 ===== #
    figs = []
    for k, layer in enumerate(data.states):

        figs2 = plt.figure(constrained_layout=True,
                          num="gates L{} (trial {})".format(k+1, data.trial))

        widths = [8]
        heights = [6, 6, 6]
        grid = figs2.add_gridspec(ncols=1, nrows=3, width_ratios=widths,
                                height_ratios=heights)

        axes2 = []
        for row in range(len(heights)):
            for col in range(len(widths)):
                axes2.append(figs2.add_subplot(grid[row, col]))

        colormap = "Blues_r"
        im = axes2[0].imshow(layer.i1[:, batch, :].T,
                             cmap = colormap,
                             aspect="auto")
        axes2[0].set_title("input gate (L{})".format(k+1))
        axes2[0].set_ylabel('unit idx')
        axes2[0].set_xticks(data.step)
        figs2.colorbar(im, ax=axes2[0])

        im = axes2[1].imshow(layer.f1[:, batch, :].T,
                             cmap = colormap,
                             aspect="auto")
        axes2[1].set_title('forget gate (L{})'.format(k+1))
        axes2[1].set_xticks(data.step)
        axes2[1].set_ylabel('unit idx')
        figs2.colorbar(im, ax=axes2[1])

        im = axes2[2].imshow(layer.o1[:, batch, :].T,
                             cmap = colormap,
                             aspect="auto")
        axes2[2].set_title('output gate (L{})'.format(k+1))
        axes2[2].set_ylabel('unit idx')
        axes2[2].set_xticks(data.step)
        figs2.colorbar(im, ax=axes2[2])

        figs.append(figs2)


    return fig1, figs2


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device = {}".format(device))
    if device == "cuda":
        logging.info("device name = {}".format(torch.cuda.get_device_name()))

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
    ds = Dataset(metadict=markers)
    ds.load_dict(path=args.vocab_file)

    # convert tokenized strings to torch.LongTensor, this will write to
    # .seq and .seq_ids attributes
    ds.tokenize_input_sequences(input_sequences=input_set, tokenizer_func=word_tokenize)
    ds.tokens_to_ids()

    # read in the configuration file which contains values for model parameters
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # initialize model class and use the custom LSTMWithStates class
    model = RNNModel(rnn_type='LSTM',
                     ntoken=config['n_vocab'],
                     ninp=config['n_inp'],
                     nhid=config['n_hid'],
                     nlayers=config['n_layers'],
                     store_states=False)

    # initialize model class and use the custom LSTMWithStates class
    #model2 = RNNModel(rnn_type='LSTMWithStates',
    #                 ntoken=config['n_vocab'],
    #                 ninp=config['n_inp'],
    #                 nhid=config['n_hid'],
    #                nlayers=config['n_layers'],
    #                 store_states=False)

    # make this input argument at some point
    print(device)
    model.load_state_dict(
        torch.load(args.model_weights, map_location=torch.device(device)))
    #model2.load_state_dict(torch.load(args.model_weights))

    model.eval()
    #model2.eval()

    # ===== EXPERIMENT CLASS ===== #
    logging.info("Running experiment...")
    logging.info("Per token output == {}".format(args.per_token_output))
    exp = Experiment(model=model.to(device), dictionary=ds.dictionary, device=device)
#exp2 = Experiment(model=model2, dictionary=ds.dictionary)

    outputs = exp.run(input_set=ds.seq_ids, metainfo=ds.meta.__dict__)
    #outputs2 = exp2.run(input_set=ds.seq_ids, metainfo=ds.meta.__dict__)

    # way to test implementations
    #sent1, sent2 = 3, 6
    #d1 = np.stack((outputs[sent1].log_probs[:, :, 0], outputs2[sent1].log_probs[:, :, 0]))
    #d2 = np.stack((outputs[sent2].log_probs[:, :, 0], outputs2[sent2].log_probs[:, :, 0]))

    # test that all values are equal, except the first, which is nan
    #assert (d1[1, 1::, 0] == d1[0, 1::, 0]).all()
    #assert (d2[1, 1::, 0] == d2[0, 1::, 0]).all()

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
                 'log_probs': o.log_probs,
                 'states': [SimpleNamespace(**vars(s)) if s else [] for s in o.states],}
                  for o in outputs]

    # convert dict to a SimpleNamespace to have dot indexing instead of brackets
    data = [SimpleNamespace(**el) for el in datadict]

    # wrapper
    def datanamespace_to_df(data):

        dfs = []

        for el in data:
            
            # if per token output is to be saved, create appropriate columns in data frame
            if args.per_token_output:
                cols = ["word", "corpuspos", "markers", "surp"]

                input_arrays = [np.array(el.token),
                                el.step,
                                el.markers,
                                el.log_probs[:, 0, 0]]

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
    
    # save output as csv
    savename = os.path.join(args.output_folder, args.output_filename)

    logging.info('Saving {}'.format(savename))
    df.to_csv(savename)

