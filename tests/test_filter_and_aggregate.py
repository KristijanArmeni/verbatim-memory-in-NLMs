
import pandas as pd
import numpy as np
from src.wm_suite.viz.func import filter_and_aggregate


def generate_sequence(tokens, val1, val2):

    tokens = ["A", "dog", "says", ":"] + tokens + [".", "A", "cat", "says", ":"] + tokens + ["."]
    first_list_vals = np.repeat(val1, 3).tolist()
    second_list_vals = np.repeat(val2, 3).tolist()
    values1 = np.random.uniform(0, 10, [4]).tolist() + first_list_vals + np.random.uniform(0, 10, 5).tolist() + second_list_vals + [np.random.uniform(0, 10)]
    
    return tokens, values1


def test_filter_and_aggregate():

    t1, v1 = generate_sequence(["door", "roof", "candle"], 8., 4.)
    t2, v2 = generate_sequence(["car", "shop", "window"], 8., 8.)
    t3, v3 = generate_sequence(["house", "spoon", "north"], 8., 0.)

    stimid = np.repeat([1, 2, 3], len(t1))
    marker = np.tile([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3], 3)
    marker_pos_rel = np.tile([-3, -2, -1, np.nan, 0, 1, 2, np.nan, -4, -3, -2, -1, 0, 1, 2, np.nan], 3)

    rec = {"token": t1 + t2 + t3, "surp": v1 + v2 + v3, "stimid": stimid, "marker": marker, "marker_pos_rel": marker_pos_rel}

    df = pd.DataFrame(rec)

    # add other expected columns in df

    df['model'] = "gpt2"
    df['second_list'] = "repeat"
    df['list'] = "random"
    df['prompt_len'] = 8
    df["list_len"] = 3
    df['context'] = "intact"
    df['model_id'] = "test"

    # grad the first time step (technically, mean computation is applied, but no effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [0, 1, 2]}]

    # run the tested function
    df2, dagg_ = filter_and_aggregate(df, 
                                    model="gpt2", 
                                    model_id=df.model_id.unique().item(), 
                                    groups=variables, 
                                    aggregating_metric="mean")

    repeat_surp = df2.x_perc.to_numpy()

    # check that repeat surprisals are as expected
    assert repeat_surp[0] == 50
    assert repeat_surp[1] == 100
    assert repeat_surp[2] == 0

    # grad the first time step (technically, mean computation is applied, but no effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [0]}]

    # run the tested function
    df2, dagg_ = filter_and_aggregate(df, 
                                    model="gpt2", 
                                    model_id=df.model_id.unique().item(), 
                                    groups=variables, 
                                    aggregating_metric="mean")

    repeat_surp = df2.x_perc.to_numpy()

    # check that repeat surprisals are as expected
    assert repeat_surp[0] == 50
    assert repeat_surp[1] == 100
    assert repeat_surp[2] == 0
