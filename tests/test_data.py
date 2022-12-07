import json
import pytest
import torch
from stimuli import prompts, prefixes
from prepare_transformer_inputs import concat_and_tokenize_inputs
from wm_suite.experiment import Dataset, read_marker_file

from models.awd_lstm.model import RNNModel as AWD_RNNModel
from models.awd_lstm.splitcross import SplitCrossEntropyLoss

from transformers import GPT2TokenizerFast
from nltk import word_tokenize


@pytest.fixture
def transformer_test_data():

    # make some input sentences
    test_lists = [["window", "cannon", "apple"], 
                ["village", "shipping", "beauty"],
                ["resort", "rival", "village"],
                ["research", "resort", "rival"],
                ["lumber", "research", "resort"],
                ]

    test_inputs, metadata = concat_and_tokenize_inputs(prefix=prefixes["sce1"]["1"],
                                             prompt=prompts["sce1"]["1"],
                                             word_list1=test_lists,
                                             word_list2=test_lists,
                                             ngram_size="3",
                                             tokenizer=GPT2TokenizerFast.from_pretrained("gpt2"),
                                             bpe_split_marker="Ġ",
                                             marker_logic="outside",
                                             ismlm=False)


    return test_inputs, metadata


@pytest.fixture
def awd_lstm_dataset():

    n_inputs = 10

    list_fname = "/home/ka2773/project/lm-mem/src/data/rnn_input_files/categorized_lists_sce1-test_control.txt"
    markers_fname = "/home/ka2773/project/lm-mem/src/data/rnn_input_files/categorized_lists_sce1-test_control_markers.txt"

    with open(list_fname, "r") as f:
        input_set = [l.strip("\n").lower() for l in f.readlines()][0:n_inputs]

    markers_tmp = read_marker_file(markers_fname)

    markers = {key: markers_tmp[key][0:10] for key in markers_tmp.keys()}

    ds = Dataset(metadict=markers)
    ds.dictionary = torch.load("models/awd_lstm/data/wikitext-103/dictionary")

    ds.tokenize_input_sequences(input_sequences=input_set, tokenizer_func=word_tokenize)
    ds.tokens_to_ids()

    return ds


@pytest.fixture
def awd_lstm_weights_and_criterion():

    fn = "/scratch/ka2773/project/lm-mem/checkpoints/awd_lstm/AWD-LSTM_3-layer_config.json"

    with open(fn, "r") as fh:
        config = json.load(fh)

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

        # instantiate the loss
    splits = [2800, 20000, 76000]  # these are hard-coded
    criterion = SplitCrossEntropyLoss(config["n_hid"], splits, verbose=False)

    model, criterion, _ = torch.load("/scratch/ka2773/project/lm-mem/checkpoints/awd_lstm/LSTM_3-layer_adam.pt", map_location='cpu') 

    return model, criterion