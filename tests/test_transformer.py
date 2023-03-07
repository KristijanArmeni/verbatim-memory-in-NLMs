
from typing import Dict
import torch
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from test_data import transformer_test_data
from src.wm_suite.wm_test_suite import Experiment

def test_transformer_experiment(transformer_test_data):

    # load model and its tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # set to evaluation mode
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #initialize experiment class
    experiment = Experiment(model=model, 
                            ismlm=False,
                            tokenizer=tokenizer,
                            context_len=1024,
                            batch_size=1,
                            stride=1,
                            use_cache=False,
                            device=device)


    # ===== RUN EXPERIMENT LOOP ===== #
    inputs, metadata = transformer_test_data
    output_dict = experiment.start(input_sequences = inputs)

    n_stim = len(inputs)

    # check markers, should be only values 0, 1, 2, 3
    assert np.array_equal(np.unique(metadata["trialID"][0]), [0, 1, 2, 3])

    # check that there are 4 subsequences in positionID field, they start with 0-index
    assert len(np.where(np.array(metadata["positionID"][0]) == 0)[0]) == 4

    assert isinstance(output_dict, Dict)

    # all log likelihoods are non-negative
    assert np.all([np.all(np.array(output_dict['surp'][i])[1::] > 0) for i in range(n_stim)])