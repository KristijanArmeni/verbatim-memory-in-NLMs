import json
import pytest
import torch
from wm_suite.io.stimuli import prompts, prefixes
from wm_suite.io.prepare_transformer_inputs import concat_and_tokenize_inputs
from wm_suite.experiment import Dataset, read_marker_file

from models.awd_lstm.model import RNNModel as AWD_RNNModel
from models.awd_lstm.splitcross import SplitCrossEntropyLoss

from transformers import GPT2TokenizerFast, AutoTokenizer
from nltk import word_tokenize


@pytest.fixture
def transformer_test_data():

    # make some input sentences
    test_lists = [["window", "cannon", "apple"], 
                ["village", "shipping", "beauty"],
                ]

    test_inputs = concat_and_tokenize_inputs(prefix=prefixes["sce1"]["1"],
                                             prompt=prompts["sce1"]["1"],
                                             word_list1=test_lists,
                                             word_list2=test_lists,
                                             ngram_size="3",
                                             tokenizer=GPT2TokenizerFast.from_pretrained("gpt2"))


    return test_inputs

@pytest.fixture
def transformer_test_data_with_split_token():

    # make some input sentences
    # we know that <prairie> will be split into <pra-irie>
    # by gpt2 and pythia tokenizers
    test_lists = [["window", "cannon", "apple"], 
                 ["village", "prairie", "beauty"],
                 ]

    test_inputs_gpt2 = concat_and_tokenize_inputs(prefix=prefixes["sce1"]["1"],
                                             prompt=prompts["sce1"]["1"],
                                             word_list1=test_lists,
                                             word_list2=test_lists,
                                             ngram_size="3",
                                             tokenizer=GPT2TokenizerFast.from_pretrained("gpt2"))

    test_inputs_pythia = concat_and_tokenize_inputs(prefix=prefixes["sce1"]["1"],
                                             prompt=prompts["sce1"]["1"],
                                             word_list1=test_lists,
                                             word_list2=test_lists,
                                             ngram_size="3",
                                             tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m"))


    return test_inputs_gpt2, test_inputs_pythia


@pytest.fixture
def transformer_wt103_test_data():

    # make some input sentences
    test_lists = [["window", "cannon", "apple"], 
                ["village", "shipping", "beauty"],
                ]

    test_inputs = concat_and_tokenize_inputs(prefix=prefixes["sce1"]["1"],
                                             prompt=prompts["sce1"]["1"],
                                             word_list1=test_lists,
                                             word_list2=test_lists,
                                             ngram_size="3",
                                             tokenizer=GPT2TokenizerFast.from_pretrained("gpt2"))


    return test_inputs


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
    ds.dictionary = torch.load("/home/ka2773/project/lm-mem/src/models/awd_lstm/data/wikitext-103/dictionary")

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

    _, criterion, _ = torch.load("/scratch/ka2773/project/lm-mem/checkpoints/awd_lstm/LSTM_3-layer_adam.pt", map_location='cpu') 

    model.load_state_dict(torch.load("/scratch/ka2773/project/lm-mem/checkpoints/awd_lstm/LSTM_3-layer_adam_statedict.pt", map_location='cpu'))


    return model, criterion
