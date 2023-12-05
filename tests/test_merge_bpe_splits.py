from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    GPT2Config,
    AutoTokenizer,
)
from wm_suite.wm_test_suite import Experiment
from test_data import transformer_test_data_with_split_token

def test_merge_bpe_splits(transformer_test_data_with_split_token):
    """Run experiment on two models with two tokenizers
    and check that the final merged strings are equal irrespective
    of the tokenizer idionsyncracies.
    """
    # load model and its tokenizer
    model_gpt2 = GPT2LMHeadModel(GPT2Config(n_layer=3, n_head=4, n_embd=128))
    tokenizer_gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")

    model_pythia = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", revision="main"
    )
    tokenizer_pythia = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    model_gpt2.eval()
    model_pythia.eval()

    # run experiments with two models
    experiment1 = Experiment(
        model=model_gpt2,
        ismlm=False,
        tokenizer=tokenizer_gpt2,
        context_len=1024,
        batch_size=1,
        stride=1,
        use_cache=False,
        device="cpu",
    )

    experiment2 = Experiment(
        model=model_pythia,
        ismlm=False,
        tokenizer=tokenizer_pythia,
        context_len=1024,
        batch_size=1,
        stride=1,
        use_cache=False,
        device="cpu",
    )

    inputs_gpt2, inputs_pythia = transformer_test_data_with_split_token

    output_dict_gpt2 = experiment1.start(
        input_sequences=[sequence.ids for sequence in inputs_gpt2]
    )
    output_dict_pythia = experiment2.start(
        input_sequences=[sequence.ids for sequence in inputs_pythia]
    )

    # additional token information to dicts
    output_dict_gpt2["token"] = [s.toks.copy() for s in inputs_gpt2]
    output_dict_gpt2["subtok_ids"] = [s.subtok_ids.copy() for s in inputs_gpt2]
    output_dict_gpt2["trial_ids"] = [s.trial_ids.copy() for s in inputs_gpt2]

    output_dict_pythia["token"] = [s.toks.copy() for s in inputs_pythia]
    output_dict_pythia["subtok_ids"] = [s.subtok_ids.copy() for s in inputs_pythia]
    output_dict_pythia["trial_ids"] = [s.trial_ids.copy() for s in inputs_pythia]

    # now merge the splits
    output_dict_gpt2_merged = experiment1.merge_bpe_splits(outputs=output_dict_gpt2)
    output_dict_pythia_merged = experiment2.merge_bpe_splits(outputs=output_dict_pythia)

    # now check that the two strings are equal after merging
    gpt2_toks = output_dict_gpt2_merged["token"].copy()
    pythia_toks = output_dict_pythia_merged["token"].copy()

    gpt2_reconstructed_input = " ".join([e.strip(" ") for e in gpt2_toks[0]])
    pythia_reconstructed_input = " ".join([e.strip(" ") for e in pythia_toks[0]])

    assert (
        gpt2_reconstructed_input == pythia_reconstructed_input
    ), "Merged strings are not equal"
