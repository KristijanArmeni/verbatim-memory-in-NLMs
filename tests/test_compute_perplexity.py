import torch
from numpy.testing import assert_allclose
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from test_data import transformer_test_data
from wm_suite.wm_test_suite import Experiment
from models.transformer.train_gpt2_ import compute_perplexity
from wm_suite.utils import set_cuda_if_available

#from wm_suite.io.wt103.dataset import WikiTextDataset


def test_compute_perplexity(transformer_test_data):

    # load a tiny random model and its tokenizer
    model = GPT2LMHeadModel(GPT2Config(n_layer=3, n_head=4, n_embd=32))
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # set to evaluation mode
    model.eval()

    device = set_cuda_if_available("cuda")

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
                            batch_size=2,
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
        ppl2.append(experiment.ppl(input_ids=inp.ids.unsqueeze(0).to(device),
                                   context_len=1024, stride=1)[0])

    # using compute_perplexity()
    ppl = []
    for inp in inputs:
        ppl.append(compute_perplexity(model=model, 
                                     input_ids=inp.ids.unsqueeze(0), 
                                     tokenizer=tokenizer, 
                                     context_len=1024, 
                                     stride=1, 
                                     device=device)[0])

    # check that all give the same result to at least 2 decimal places
    values1 = output_dict1['sequence_ppl']
    values2 = output_dict2['sequence_ppl']

    assert_allclose(values1, values2, rtol=1e-6)

    values1 = output_dict1['sequence_ppl']
    values2 = ppl
    assert_allclose(values1, values2, rtol=1e-6)

    values1 = ppl
    values2 = ppl2
    assert_allclose(values1, values2, rtol=1e-6)
