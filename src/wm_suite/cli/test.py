from typing import Dict
import torch
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from wm_suite.io.test_ds import get_test_data
from wm_suite.wm_test_suite import Experiment
import logging


def test_install():
    logging.info("Testing installation by doing a short test run...")

    # load model and its tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # set to evaluation mode
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize experiment class
    experiment = Experiment(
        model=model,
        ismlm=False,
        tokenizer=tokenizer,
        context_len=1024,
        batch_size=1,
        stride=1,
        use_cache=False,
        device=device,
    )

    # ===== RUN EXPERIMENT LOOP ===== #
    inputs = get_test_data()
    output_dict = experiment.start(
        input_sequences=[sequence.ids for sequence in inputs]
    )

    logging.info("Run complete! Installation successful!")

    return 0
