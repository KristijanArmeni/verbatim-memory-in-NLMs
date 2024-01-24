import numpy as np
from wm_suite.io.sample_noun_lists import chunk_lists_and_circular_shift


def test_chunk_lists_and_circular_shift():
    # set up a small list of nouns
    nouns = [
        "whisky",
        "water",
        "vase",
        "umbrella",
        "turtle",
        "tree",
        "tractor",
        "toothbrush",
        "tomato",
        "toad",
        "tiger",
        "telescope",
        "tablespoon",
        "spaghetti",
        "saxophone",
    ]

    out = chunk_lists_and_circular_shift(nouns, list_length=3)

    assert len(out["n3"]) == len(nouns)

    # test that each noun occurs in all positions (test first 3 nouns)
    assert np.array_equal(out["n3"][0], ["whisky", "water", "vase"])
    assert np.array_equal(out["n3"][1], ["vase", "whisky", "water"])
    assert np.array_equal(out["n3"][2], ["water", "vase", "whisky"])

    # test that it works for nouns from the middle of the list
    assert np.array_equal(out["n3"][6], ["tractor", "toothbrush", "tomato"])
    assert np.array_equal(out["n3"][7], ["tomato", "tractor", "toothbrush"])
    assert np.array_equal(out["n3"][8], ["toothbrush", "tomato", "tractor"])

    # test that it works for nouns from the end of the list
    assert np.array_equal(out["n3"][-3], ["tablespoon", "spaghetti", "saxophone"])
    assert np.array_equal(out["n3"][-2], ["saxophone", "tablespoon", "spaghetti"])
    assert np.array_equal(out["n3"][-1], ["spaghetti", "saxophone", "tablespoon"])

    # test a chunk length that doesn't fit evenly into the list length
    list_len = 7
    out = chunk_lists_and_circular_shift(nouns, list_length=list_len)

    n_total_lists = len(nouns) - (len(nouns) % list_len)
    assert len(out["n7"]) == n_total_lists

    # test that each noun occurs in all positions test the first chunk
    assert np.array_equal(
        out["n7"][0],
        ["whisky", "water", "vase", "umbrella", "turtle", "tree", "tractor"],
    )
    assert np.array_equal(
        out["n7"][1],
        ["tractor", "whisky", "water", "vase", "umbrella", "turtle", "tree"],
    )
    assert np.array_equal(
        out["n7"][2],
        ["tree", "tractor", "whisky", "water", "vase", "umbrella", "turtle"],
    )
    assert np.array_equal(
        out["n7"][3],
        ["turtle", "tree", "tractor", "whisky", "water", "vase", "umbrella"],
    )
    assert np.array_equal(
        out["n7"][4],
        ["umbrella", "turtle", "tree", "tractor", "whisky", "water", "vase"],
    )
    assert np.array_equal(
        out["n7"][5],
        ["vase", "umbrella", "turtle", "tree", "tractor", "whisky", "water"],
    )
    assert np.array_equal(
        out["n7"][6],
        ["water", "vase", "umbrella", "turtle", "tree", "tractor", "whisky"],
    )
