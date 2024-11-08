from typing import Dict

import numpy as np
from test_data import transformer_wt103_test_data
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from wm_suite.utils import set_cuda_if_available
from wm_suite.wm_test_suite import Experiment


def test_transformer_wt103_experiment(transformer_wt103_test_data):
    # load model and its tokenizer
    model = GPT2LMHeadModel.from_pretrained("Kristijan/gpt2_wt103_12-layer")
    tokenizer = GPT2TokenizerFast.from_pretrained("Kristijan/wikitext-103_tokenizer_v2")

    # set to evaluation mode
    model.eval()

    device = set_cuda_if_available("cuda")

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
    inputs = transformer_wt103_test_data
    output_dict = experiment.start(
        input_sequences=[sequence.ids for sequence in inputs]
    )

    n_stim = len(inputs)

    # check markers, should be only values 0, 1, 2, 3
    assert np.array_equal(np.unique(inputs[0].trial_ids), [-1, 0, 1, 2, 3])

    # check that there are 4 subsequences in positionID field, they start with 0-index
    # assert len(np.where(np.array(inputs[0].position_ids) == 0)[0]) == 4

    assert isinstance(output_dict, Dict)

    # all log likelihoods are non-negative
    assert np.all(
        [np.all(np.array(output_dict["surp"][i])[1::] > 0) for i in range(n_stim)]
    )

    # compute perplexity on a WT103 test set
    # test_set_path = os.path.join(DATA_PATH, "wikitext-103", "wiki.test.tokens")
    # logger.info(f"Loading {test_set_path}...")
    # _, ids = WikiTextDataset(tokenizer=tokenizer).retokenize_txt(test_set_path)

    # initialize experiment class
    # experiment2 = Experiment(model=model, ismlm=False,
    #                        tokenizer=tokenizer,
    #                        context_len=1024,
    #                        batch_size=1,
    #                        stride=1,
    #                        use_cache=False,
    #                        device=device)

    # ppl, _, _ = experiment2.ppl(input_ids=torch.tensor([ids]), context_len=1024, stride=256)

    # check if WT103 perplexity is close to 40.6
    # assert np.isclose(ppl, 40.6146, atol=1e-2)
