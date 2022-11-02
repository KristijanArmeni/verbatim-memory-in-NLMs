import pytest

from stimuli import prompts, prefixes
from prepare_transformer_inputs import concat_and_tokenize_inputs

from transformers import GPT2TokenizerFast

@pytest.fixture
def transformer_test_data():

    # make some input sentences
    test_lists = [["window", "cannon", "apple"], 
                ["village", "shipping", "beauty"],
                ["resort", "rival", "village"],
                ["research", "resort", "rival"],
                ["lumber", "research", "resort"],
                ["ticket", "baby", "treasure"],
                ["marriage", "ticket", "baby"], 
                ["summer", "bottom", "meaning"],
                ["stanza", "summer", "bottom"],
                ["hatred", "stanza", "summer"],]

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
