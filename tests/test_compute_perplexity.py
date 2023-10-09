from typing import Dict
import torch
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from test_data import transformer_test_data
from wm_suite.wm_test_suite import Experiment
from models.transformer.train_gpt2_ import compute_perplexity
import logging

from data.wt103.dataset import WikiTextDataset

def test_compute_perplexity(transformer_test_data):

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

    #initialize experiment class, but batch size at 5
    experiment2 = Experiment(model=model, 
                            ismlm=False,
                            tokenizer=tokenizer,
                            context_len=1024,
                            batch_size=5,
                            stride=1,
                            use_cache=False,
                            device=device)


    # ===== RUN EXPERIMENT LOOP ===== #
    inputs = transformer_test_data
    
    # using.ppl_batched()
    output_dict1 = experiment.start(input_sequences = [e.ids for e in inputs])   # batch size at 1
    output_dict2 = experiment2.start(input_sequences = [e.ids for e in inputs])  # batch size at 5

    # using experiment.ppl()
    ppl2 = []
    for inp in inputs:
        ppl2.append(experiment.ppl(input_ids=inp.ids, context_len=1024, stride=1)[0].item())

    # using compute_perplexity()
    ppl = []
    for inp in inputs:
        ppl.append(compute_perplexity(model=model, 
                                     input_ids=inp.ids, 
                                     tokenizer=tokenizer, 
                                     context_len=1024, 
                                     stride=1, 
                                     device=device)[0].item())

    # check that all give the same result to at least 3 decimal places
    values1 = output_dict1['sequence_ppl']
    values2 = output_dict2['sequence_ppl']
    assert all([np.isclose(v1, v2, rtol=0, atol=1e-3) for v1, v2 in zip(values1, values2)])

    values1 = output_dict1['sequence_ppl']
    values2 = ppl
    assert all([np.isclose(v1, v2, rtol = 0, atol=1e-3) for v1, v2 in zip(values1, values2)])

    values1 = ppl
    values2 = ppl2
    logging.info(values1)
    logging.info(values2)
    assert all([np.isclose(v1, v2, rtol = 0, atol=1e-3) for v1, v2 in zip(values1, values2)])
