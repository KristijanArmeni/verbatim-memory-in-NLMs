import numpy as np
from src.wm_suite.io.prepare_transformer_inputs import mark_subtoken_splits

def test_mark_subtoken_splits():

    # this is the "outside" logic
    tokens = ["<|endoftext|>", "ĠI", "Ġsaw", "Ġa", "Ġsp", "arrow", "Ġyesterday", "!"]
    target_ids = [-2, 1, 2, 3, 4, 4, 5, -1]

    ids = mark_subtoken_splits(tokens=tokens,
                                split_marker="Ġ",
                                marker_logic="outside",
                                eos_markers=["<|endoftext|>", "<|endoftext|>"])

    # we should get the coding above
    assert np.all(ids == target_ids)

    # this is the 'inside' logic
    tokens = ["<|endoftext|>", "I", "saw", "a", "sp", "##arrow", "yesterday", "!"]
    target_ids = [-2, 1, 2, 3, 4, 4, 5, -1]

    ids = mark_subtoken_splits(tokens=tokens,
                                split_marker="##",
                                marker_logic="within",
                                eos_markers=["<|endoftext|>", "<|endoftext|>"])

    assert np.all(ids == target_ids)