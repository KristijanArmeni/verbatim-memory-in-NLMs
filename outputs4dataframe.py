
import os
import pandas as pd
import numpy as np

output_folder = "./output"

# file naming syntax:
# metric_model_scenario_condition_list-type
files_gpt = [
    "surprisal_gpt2_sce1rnd_permute_categorized.csv",
    "surprisal_gpt2_sce1rnd_repeat_categorized.csv",
    "surprisal_gpt2_sce1rnd_control_categorized.csv",
    "surprisal_gpt2_sce1rnd_permute_random.csv",
    "surprisal_gpt2_sce1rnd_repeat_random.csv",
    "surprisal_gpt2_sce1rnd_control_random.csv",
    "surprisal_gpt2_sce1_permute_categorized.csv",
    "surprisal_gpt2_sce1_repeat_categorized.csv",
    "surprisal_gpt2_sce1_control_categorized.csv",
    "surprisal_gpt2_sce1_permute_random.csv",
    "surprisal_gpt2_sce1_repeat_random.csv",
    "surprisal_gpt2_sce1_control_random.csv"
]

files_rnn = [
    "surprisal_rnn_sce1_permute_categorized.csv",
    "surprisal_rnn_sce1_repeat_categorized.csv",
    "surprisal_rnn_sce1_permute_random.csv",
    "surprisal_rnn_sce1_repeat_random.csv",
    "surprisal_rnn_sce1_control_categorized.csv",
    "surprisal_rnn_sce1_control_random.csv",
    "surprisal_rnn_sce1rnd_permute_categorized.csv",
    "surprisal_rnn_sce1rnd_repeat_categorized.csv",
    "surprisal_rnn_sce1rnd_permute_random.csv",
    "surprisal_rnn_sce1rnd_repeat_random.csv",
    "surprisal_rnn_sce1rnd_control_categorized.csv",
    "surprisal_rnn_sce1rnd_control_random.csv",
]


def load_and_preproc_csv(output_folder, filenames):

    # load output .csvs in a loop, then concatenate them
    dfs = []

    for file in filenames:

        print("Reading \n {}".format(file))

        arc = file.split("_")[1]  # "rnn" or "gpt2"

        sep = "," if arc == "gpt2" else "\t"

        dftmp = pd.read_csv(os.path.join(output_folder, file), sep=sep, header=0)

        # add information from filenames
        dftmp["list"] = file.split("_")[-1].split(".")[0]  # add column on list composition
        dftmp["second_list"] = file.split("_")[-2]  # store information on second list
        dftmp["scenario"] = file.split("_")[-3]

        # remove punctuation prior to creating token index
        # filter out punctuation
        if arc == "gpt2":
            dftmp = dftmp.loc[~dftmp.ispunct, :]

            # rename some columns to avoid the need for if/else lower
            dftmp.rename(columns={"stimID": "sentid", "trialID" : "marker", "prompt": "prompt_len"}, inplace=True)

        elif arc == "rnn":
            dftmp = dftmp.loc[~dftmp.word.isin([":", ".", ","])].copy()

        # add "sentpos" column containing token index within a sentence
        if arc == "gpt2":
            tmp = []
            for s in dftmp.sentid.unique():
                tmp.append(np.arange(0, len(dftmp.loc[dftmp.sentid == s, "sentid"])))
            dftmp["sentpos"] = np.concatenate(tmp)

        # now also create marker pos index (start indexing with 1)
        tmp = []  # clear variable
        for sentid in dftmp.sentid.unique():
            t = dftmp.loc[dftmp.sentid == sentid]
            for m in t.marker.unique():
                tmp.append(np.arange(0, len(t.loc[t.marker == m]))+1)
        dftmp["marker_pos"] = np.concatenate(tmp)

        # let's also create a column where the pre-target tokens are indexed relative
        # to the target (-3, -2, -1, ...) markers so whe can plot those as well
        tmp = []  # clear variable
        for sentid in dftmp.sentid.unique():
            t = dftmp.loc[dftmp.sentid == sentid]
            for m in t.marker.unique():
                # use relative indexing for markers prior to target tokens
                ntok = len(t.loc[t.marker == m])
                if m == 2:
                    # start with 1, reverse and make it negative
                    tmp.append(-1*((np.arange(0, ntok)+1)[::-1]))
                else:
                    tmp.append((np.arange(0, ntok)))
        dftmp["marker_pos_rel"] = np.concatenate(tmp)

        dfs.append(dftmp)
        dftmp, tmp = None, None

    return pd.concat(dfs)


#===== LOAD CSV FILES =====#
print("Preprocessing gpt output...")
gpt = load_and_preproc_csv(output_folder=output_folder, filenames=files_gpt)

print("Preprocessing rnn output...")
rnn = load_and_preproc_csv(output_folder=output_folder, filenames=files_rnn)

# fix some renaming etc

# rename prompt length values to more meaningful ones
prompt_len_map = {
    1: 8,
    2: 30,
    3: 100,
    4: 200,
    5: 400,
}
gpt.prompt_len = gpt.prompt_len.map(prompt_len_map)
rnn.prompt_len = rnn.prompt_len.map(prompt_len_map)

# rename scenario values to more meaningful ones
scenario_map = {
    "sce1": "intact",
    "sce1rnd": "scrambled",
}

gpt.scenario = gpt.scenario.map(scenario_map)
rnn.scenario = rnn.scenario.map(scenario_map)

# rename the "scenario" column to "context"
gpt.rename(columns={"scenario": "context"}, inplace=True)
rnn.rename(columns={"scenario": "context"}, inplace=True)

# drop some redundant columns creted by Pandas bookkeeping system
# rnn.drop(["Unnamed: 0"], axis=1, inplace=True)
gpt.drop(["Unnamed: 0"], axis=1, inplace=True)

# save back to csv (waste of memory but let's stick to this for now)
fname = os.path.join(output_folder, "output_gpt2.csv")
print("Saving {}".format(fname))
gpt.to_csv(fname, sep="\t")

fname = os.path.join(output_folder, "output_rnn.csv")
print("Saving {}".format(fname))
rnn.to_csv(fname, sep="\t")

