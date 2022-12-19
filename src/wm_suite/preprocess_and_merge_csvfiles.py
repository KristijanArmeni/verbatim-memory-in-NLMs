
import os
import sys
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import argparse
from string import punctuation
import logging

logging.basicConfig(format="[INFO] %(message)s", level=logging.INFO)


def infer_labels_from_filebasename(filename):
    """
    infer_labels_from_filebasename(filename)

    Parameters:
    ----------
    filename : str
        filename to surprisal_<arc>_<model-id>_<scenario>_<second_list>_<list_type>.csv (output of wm_test_suite.py) files
    
    Returns:
    -------
    arc : str
        string denoting architecture (e.g. 'gpt2')
    model_id : str
        string denoting rather arbitrary model_id (e.g. "a-10")
    scenario : str
        string denoting which scenario was used (e.g. "sce1")
    second_list : str
        string denoting the type of second list (e.g. "repeat")
    list_type : str
        string denoting whether the list is random or categorized
    """
    arc, model_id, scenario, prompt_len, list_len, condition, list_type = filename.split("_")[1::]
    list_type = list_type.split(".csv")[0]  # cut the filename suffix

    logging.info(f"Inferring the following information from filename {filename}: \n" + \
                f"list length = {list_len}\n" + \
                f"prompt length = {prompt_len}\n" + \
                f"list_type = {list_type}\n" + \
                f"condition = {condition}\n" + \
                f"scenario = {scenario}\n" + \
                f"model_id = {model_id}\n" + \
                f"architecture = {arc}")

    return arc, model_id, scenario, list_len, prompt_len, list_type, condition

def load_and_preproc_csv(output_folder, filenames):

    out = []

    for file in filenames:

        print("Reading \n {}".format(file))

        labels = infer_labels_from_filebasename(os.path.basename(file))
        arc, model_id, scenario, list_len, prompt_len, second_list, list_type = labels 

        sep = ","

        dftmp = pd.read_csv(os.path.join(output_folder, file), sep=sep, header=0)

        # add information from filenames
        dftmp["list"] = list_type  # add column on list composition
        dftmp["second_list"] = second_list  # store information on second list
        dftmp["scenario"] = scenario
        dftmp["model_id"] = model_id

        ngram_list_labels = ["ngram-random", "ngram-categorized"]

        # remove punctuation prior to creating token index
        # filter out punctuation
        if arc in ["gpt2", "bert"]:

            # rename some columns to avoid the need for if/else lower
            dftmp.rename(columns={"stimID": "sentid", "trialID" : "marker", "prompt": "prompt_len"}, inplace=True)

            # throw out punctuation and eos
            eos_symbols = ['<|endoftext|>', "[CLS]", '[SEP]']
            dftmp = dftmp.loc[(~dftmp.token.isin(list(punctuation) + eos_symbols))]

            # we need these columns in the output after merging
            columns = ["subtok", "sentid", "stimid", "list_len", "prompt_len",
                        "scenario", "list", "second_list", "model_id", "marker"]

            # ngram experiment doesn't have prompts, but distractors
            if list_type in ngram_list_labels:

                # we need these columns in the output after merging
                columns = ["subtok", "subtok_markers", "sentid", "stimid", "list_len", "prompt_len",
                           "scenario", "list", "second_list", "model_id", "marker"]


            # merge subtokens add the token markers and relative markers
            dftmp = preprocess_gpt_dataframe(dfin=dftmp.copy(), has_subtoks=True,
                                             keep_groups=columns)

            # change some column names for ngram experiment appropriately
            if list_type in ngram_list_labels:

                dftmp.rename(columns = {"prompt_len": "dist_len", "list_len": "ngram_len" }, inplace=True)

        elif np.any([l in arc for l in ['rnn', 'awd-lstm']]):

            dftmp = dftmp.loc[~dftmp.word.isin([":", ".", ","])].copy()

            # temporarily
            dftmp.rename(columns= {"markers": "marker"}, inplace = True)

            # only add the token markers and relative markers
            dftmp = preprocess_rnn_dataframe(dfin=dftmp)

            # TEMP rename column to make it consistent, consdier fixing
            # this upstream
            if list_type in ngram_list_labels:

                dftmp.rename(columns={"list_len": "ngram_len", "prompt_len": "dist_len"}, inplace=True)

        # append df for this experiment
        out.append(dftmp)

    return pd.concat(out, ignore_index=True)


def preprocess_gpt_dataframe(dfin, has_subtoks=None, keep_groups=None):

    # now do the massive loops and merge subtokens
    merged_dfs = []       # here we will store merged dfs
    marker_pos = []       # this is for marker arrays
    marker_pos_rel = []

    # loop over list conditions
    for llen in dfin.list_len.unique():

        sel1 = (dfin.list_len==llen)

        # loop over markers
        for sent in tqdm(dfin.loc[sel1].sentid.unique(), desc="list len == {}".format(llen)):

            sel2 = (dfin.list_len==llen) & (dfin.sentid==sent)

            # loop over sentences
            for mark in dfin.loc[sel2].marker.unique():

                # merge tokens for this list_len, this sentece and this  marker
                sel3 = (dfin.list_len == llen) &\
                       (dfin.sentid == sent) &\
                       (dfin.marker == mark)

                n_rows = len(dfin.loc[sel3])

                # only gpt-2 outputs have subtoks that need to be merged
                if has_subtoks and (keep_groups is not None):

                    df_merged = merge_subtoks(df=dfin.loc[sel3].copy(),
                                              group_levels=keep_groups,
                                              merge_operation="sum")

                    n_rows = len(df_merged)
                    merged_dfs.append(df_merged)

                # code markers relative to target tokens lists
                if mark in [0, 2]:
                    # start with 1, reverse and make it negative
                    marker_pos_rel.append(-1*((np.arange(0, n_rows)+1)[::-1]))
                else:
                    marker_pos_rel.append(np.arange(0, n_rows))

                # code marker position without negative indices
                marker_pos.append(np.arange(0, n_rows))

    if has_subtoks:
        dfout = pd.concat(merged_dfs)
    else:
        dfout = dfin

    dfout["marker_pos"] = np.concatenate(marker_pos)
    dfout["marker_pos_rel"] = np.concatenate(marker_pos_rel)

    return dfout


def merge_subtoks(df, group_levels, merge_operation="sum"):
    """
    helper function to perform averging over subtokens via .groupby and .agg
    """

    dfout = df.groupby(group_levels, sort=False)\
              .agg({"surp": merge_operation, "token": "_".join})\
              .reset_index()

    return dfout


def preprocess_rnn_dataframe(dfin, has_subtoks=None, keep_groups=None):

    # now do the massive loops and merge subtokens
    marker_pos = []       # this is for marker arrays
    marker_pos_rel = []

    # loop over list conditions
    for sent in tqdm(dfin.sentid.unique(), desc="sentid"):

        # loop over sentences
        for mark in dfin.loc[dfin.sentid==sent].marker.unique():

            # merge tokens for this list_len, this sentece and this  marker
            sel1 = (dfin.sentid == sent) &\
                   (dfin.marker == mark)

            n_rows = len(dfin.loc[sel1])

            # code markers relative to target tokens lists
            if mark in [0, 2]:
                # start with 1, reverse and make it negative
                marker_pos_rel.append(-1*((np.arange(0, n_rows)+1)[::-1]))
            else:
                marker_pos_rel.append(np.arange(0, n_rows))

            # code marker position without negative indices
            marker_pos.append(np.arange(0, n_rows))

    dfout = dfin.copy()

    dfout["marker_pos"] = np.concatenate(marker_pos)
    dfout["marker_pos_rel"] = np.concatenate(marker_pos_rel)

    return dfout

def main(input_arguments=None):

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="",
                        help="model_id string to be placed in the filename string")
    parser.add_argument("--arch", type=str, default="",
                        help="architecture to preprocess")
    parser.add_argument("--scenario", type=str, default="",
                        help="which scenario data to load")
    parser.add_argument("--filename_pattern", type=str, default="*")
    parser.add_argument("--suffix", type=str, default=".csv")
    parser.add_argument("--output_dir", type=str,
                        help="path for storing post-processed files")
    parser.add_argument("--output_filename", type=str, default="output_merged.csv")


    # uncomment this for testing
    input_arguments = ["--output_dir", "/home/ka2773/project/lm-mem/test", "--output_filename", "test_merge.csv"]

    if input_arguments is None:
        argins = parser.parse_args()
    else:
        argins = parser.parse_args(input_arguments)

    # read in files
    filename = argins.filename_pattern + argins.suffix
    files = glob.glob(os.path.join(argins.output_dir, filename))

    if not files:
        raise Exception("Can find any files that match pattern: {}".format(os.path.join(argins.output_dir, filename)))

    files.sort()

    print("Preprocessing {}_{}_{} output...".format(argins.arch, argins.model_id, argins.scenario))
    df = load_and_preproc_csv(output_folder=argins.output_dir, filenames=files)

    # rename prompt length values to more meaningful ones
    prompt_len_map = {
        1: 8,
        2: 30,
        3: 100,
        4: 200,
        5: 400,
    }

    df.prompt_len = df.prompt_len.map(prompt_len_map)

    # rename scenario values to more meaningful ones
    scenario_map = {
        "sce1": "intact",
        "sce1rnd": "scrambled",
        "sce2": "incongruent",
        "sce3": "short"
    }

    df.scenario = df.scenario.map(scenario_map)

    # rename the "scenario" column to "context"
    df.rename(columns={"scenario": "context"}, inplace=True)

    # save back to csv (waste of memory but let's stick to this for now)
    fname = os.path.join(argins.output_dir, argins.output_filename)
    print("Saving {}".format(fname))
    df.to_csv(fname, sep="\t")

#===== LOAD CSV FILES =====#
if __name__ == "__main__":

    main()
