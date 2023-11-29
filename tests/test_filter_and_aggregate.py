
import pandas as pd
import numpy as np
from src.wm_suite.viz.func import filter_and_aggregate


def generate_sequence(tokens, val1, val2):

    tokens = ["A", "dog", "says", ":"] + tokens + [".", "A", "cat", "says", ":"] + tokens + ["."]
    first_list_vals = np.repeat(val1, 3).tolist()
    second_list_vals = np.repeat(val2, 3).tolist()

    # construct sequence (sample random values in range 0, 4 on non-list tokens)
    values1 = np.random.uniform(0, 10, [4]).tolist() + first_list_vals + np.random.uniform(0, 10, 5).tolist() + second_list_vals + [np.random.uniform(0, 10)]
    
    return tokens, values1


def test_filter_and_aggregate():

    t1, v1 = generate_sequence(["door", "roof", "candle"], val1=8., val2=4.)    # value on list2 is 50% lower
    t2, v2 = generate_sequence(["car", "shop", "window"], val1=8., val2=8.)     # value on list2 is unchanged
    t3, v3 = generate_sequence(["house", "spoon", "north"], val1=8., val2=0.)   # value on list2 is at 0% lower

    # create other variables expected in the dataframe
    stimid = np.repeat([1, 2, 3], len(t1))
    marker = np.tile([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3], 3)
    marker_pos_rel1 = np.tile([-3, -2, -1, np.nan, 0, 1, 2, np.nan, 3, 4, 5, 6, 7, 8, 9, np.nan], 3)
    marker_pos_rel2 = np.tile([-10, -9, -8, np.nan, -7, -6, -5, np.nan, -4, -3, -2, -1, 0, 1, 2, np.nan], 3)

    rec = {"token": t1 + t2 + t3, "surp": v1 + v2 + v3, "stimid": stimid, "marker": marker,
           "marker_pos_rel1": marker_pos_rel1, "marker_pos_rel2": marker_pos_rel2}
    df = pd.DataFrame(rec)

    # add other expected columns in df
    df['model'] = "gpt2"
    df['second_list'] = "repeat"
    df['list'] = "random"
    df['prompt_len'] = 8
    df["list_len"] = 3
    df['context'] = "intact"

    # grad the first time step (technically, mean computation is applied, but no effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [0, 1, 2]}]

    # run the tested function
    data1, dagg_ = filter_and_aggregate(df, 
                                        independent_var="list_len",
                                        list_len_val=[3],
                                        prompt_len_val=[8],
                                        context_val=["intact"],
                                        list_positions=[0, 1, 2],
                                        aggregating_metric="mean")

    repeat_surp = data1.x_perc.to_numpy()

    # check that repeat surprisals are as expected
    assert repeat_surp[0] == 50
    assert repeat_surp[1] == 100
    assert repeat_surp[2] == 0

    # grad the first time step (technically, mean computation is applied, but not effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [0]}]

    # run the tested function
    data2, dagg_ = filter_and_aggregate(df, 
                                        independent_var="list_len",
                                        list_len_val=[3],
                                        prompt_len_val=[8],
                                        context_val=["intact"],
                                        list_positions=[0],
                                        aggregating_metric="mean")

    repeat_surp = data2.x_perc.to_numpy()

    # check that repeat surprisals are as expected
    assert repeat_surp[0] == 50
    assert repeat_surp[1] == 100
    assert repeat_surp[2] == 0


    ##### ===== TEST WHEN VALUES ARE VARIABLE PER SEQUENCE POSITOINS ===== #####

    t1, v1 = generate_sequence(["door", "roof", "candle"], val1=8., val2=4.)    # value on list2 is 50% lower
    t2, v2 = generate_sequence(["car", "shop", "window"], val1=8., val2=8.)     # value on list2 is unchanged
    t3, v3 = generate_sequence(["house", "spoon", "north"], val1=8., val2=0.)   # value on list2 is at 0% lower

    # now make sure that values at different token positions are different
    v1[-4], v1[-3], v1[-2] = 4, 0, 2
    v2[-4], v2[-3], v2[-2] = 2, 0, 4
    v3[-4], v3[-3], v3[-2] = 0, 4, 2

    # create other variables expected in the dataframe
    stimid = np.repeat([1, 2, 3], len(t1))
    marker = np.tile([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3], 3)
    marker_pos_rel1 = np.tile([-3, -2, -1, np.nan, 0, 1, 2, np.nan, 3, 4, 5, 6, 7, 8, 9, np.nan], 3)
    marker_pos_rel2 = np.tile([-10, -9, -8, np.nan, -7, -6, -5, np.nan, -4, -3, -2, -1, 0, 1, 2, np.nan], 3)

    rec = {"token": t1 + t2 + t3, "surp": v1 + v2 + v3, "stimid": stimid, "marker": marker, 
           "marker_pos_rel1": marker_pos_rel1, "marker_pos_rel2": marker_pos_rel2}
    df2 = pd.DataFrame(rec)

    # add other expected columns in df
    df2['model'] = "gpt2"
    df2['second_list'] = "repeat"
    df2['list'] = "random"
    df2['prompt_len'] = 8
    df2["list_len"] = 3
    df2['context'] = "intact"

    # grad the first time step (technically, mean computation is applied, but not effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [0]}]
    
    # run the tested function
        # run the tested function
    data1, dagg_ = filter_and_aggregate(df2, 
                                        independent_var="list_len",
                                        list_len_val=[3],
                                        prompt_len_val=[8],
                                        context_val=["intact"],
                                        list_positions=[0],
                                        aggregating_metric="mean")

    repeat_surp = data1.x_perc.to_numpy()

    # check that repeat surprisals are as expected
    assert repeat_surp[0] == 50
    assert repeat_surp[1] == 25
    assert repeat_surp[2] == 0


    # aggregate over second noun in the list (expected 0%, 0%, and 50%)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [1]}]
    
    # run the tested function
    data2, dagg_ = filter_and_aggregate(df2, 
                                        independent_var="list_len",
                                        list_len_val=[3],
                                        prompt_len_val=[8],
                                        context_val=["intact"],
                                        list_positions=[1],
                                        aggregating_metric="mean")

    repeat_surp = data2.x_perc.to_numpy()

    assert repeat_surp[0] == 0
    assert repeat_surp[1] == 0
    assert repeat_surp[2] == 50

    # aggregate over the third noun in the list (expected 0%, 0%, and 50%)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [2]}]
    
    # run the tested function
    data3, dagg_ = filter_and_aggregate(df2, 
                                        independent_var="list_len",
                                        list_len_val=[3],
                                        prompt_len_val=[8],
                                        context_val=["intact"],
                                        list_positions=[2],
                                        aggregating_metric="mean")

    repeat_surp = data3.x_perc.to_numpy()

    assert repeat_surp[0] == 25
    assert repeat_surp[1] == 50
    assert repeat_surp[2] == 25
