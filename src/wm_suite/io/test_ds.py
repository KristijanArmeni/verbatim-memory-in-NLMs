
from wm_suite.io.stimuli import prompts, prefixes
from wm_suite.io.prepare_transformer_inputs import concat_and_tokenize_inputs
from transformers import GPT2TokenizerFast
import logging

# make some input sentences
def get_test_data():

    logging.info("Generating N = 5 sample test data...")


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