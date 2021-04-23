
import os
import sys
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm


def load_and_preproc_csv(output_folder, filenames):

    out = []

    for file in filenames:

        print("Reading \n {}".format(file))

        arc = file.split("_")[1]  # "rnn" or "gpt2"

        sep = "," if arc == "gpt2" else "\t"

        dftmp = pd.read_csv(os.path.join(output_folder, file), sep=sep, header=0)

        # add information from filenames
        dftmp["list"] = file.split("_")[-1].split(".")[0]  # add column on list composition
        dftmp["second_list"] = file.split("_")[-2]  # store information on second list
        dftmp["scenario"] = file.split("_")[-3]
        dftmp["model_id"] = file.split("_")[-4]

        # remove punctuation prior to creating token index
        # filter out punctuation
        if arc == "gpt2":
                         
            # rename some columns to avoid the need for if/else lower
            dftmp.rename(columns={"stimID": "sentid", "trialID" : "marker", "prompt": "prompt_len"}, inplace=True)
            
            # throw out punctuation and eos
            dftmp = dftmp.loc[(~dftmp.ispunct) & (dftmp.token != '<|endoftext|>')]
            
            # we need these columns in the output after merging
            columns = ["subtok", "sentid", "list_len", "prompt_len", 
                        "scenario", "list", "second_list", "model_id", "marker"]
            
            # ngram experiment doesn't have prompts, but distractors
            if dftmp.list.unique() == "ngram-random":
                
                # we need these columns in the output after merging
                columns = ["subtok", "subtok_markers", "sentid", "list_len", "prompt_len", 
                          "scenario", "list", "second_list", "model_id", "marker"]

            
            # merge subtokens add the token markers and relative markers
            dftmp = preprocess_gpt_dataframe(dfin=dftmp.copy(), has_subtoks=True,
                                          keep_groups=columns)
            
            # change some column names for ngram experiment appropriately
            if dftmp.list.unique() == "ngram-random":
                
                dftmp.rename(columns = {"prompt_len": "dist_len", "list_len": "ngram_len" }, inplace=True)

        elif arc == "rnn":
            
            dftmp = dftmp.loc[~dftmp.word.isin([":", ".", ","])].copy()
            
            dftmp.rename(columns= {"markers": "marker"}, inplace = True), 
            
            # only add the token markers and relative markers
            dftmp = preprocess_rnn_dataframe(dfin=dftmp)
            
            # TEMP rename column to make it consistent, consdier fixing
            # this upstream
            if dftmp.list.unique() == "ngram-random":
                
                dftmp.rename(columns={"list_len": "ngram_len"}, inplace=True)
        
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
                                              group_levels=keep_groups)
                    
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


def merge_subtoks(df, group_levels):
    """
    helper function to perform averging over subtokens via .groupby and .agg
    """
    
    dfout = df.groupby(group_levels, sort=False)\
              .agg({"surp": "mean", "token": "_".join})\
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

    dfout = dfin
        
    dfout["marker_pos"] = np.concatenate(marker_pos)
    dfout["marker_pos_rel"] = np.concatenate(marker_pos_rel)
    
    return dfout

#===== LOAD CSV FILES =====#

if "win" in sys.platform:
    output_folder = os.path.join(os.environ['homepath'], "project", "lm-mem", 
                                 "src", "output")
elif "linux" in sys.platform:
    output_folder = os.path.join(os.environ['HOME'], "code", "lm-mem", "output")

# file naming syntax:
# metric_model_scenario_condition_list-type
files_gpt = glob.glob(os.path.join(output_folder, "surprisal_gpt2_?-*.csv"))
files_rnn = glob.glob(os.path.join(output_folder, "surprisal_rnn_?-*.csv"))

files_gpt.sort()
files_rnn.sort()

print("Preprocessing rnn output...")
rnn = load_and_preproc_csv(output_folder=output_folder, filenames=files_rnn[0:2])

print("Preprocessing gpt output...")
gpt = load_and_preproc_csv(output_folder=output_folder, filenames=files_gpt)

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
    "sce2": "incongruent",
    "sce3": "short"
}

gpt.scenario = gpt.scenario.map(scenario_map)
rnn.scenario = rnn.scenario.map(scenario_map)

# rename the "scenario" column to "context"
gpt.rename(columns={"scenario": "context"}, inplace=True)
rnn.rename(columns={"scenario": "context"}, inplace=True)

# drop some redundant columns creted by Pandas bookkeeping system
# gpt.drop(["Unnamed: 0"], axis=1, inplace=True)

# save back to csv (waste of memory but let's stick to this for now)
fname = os.path.join(output_folder, "output_gpt2.csv")
print("Saving {}".format(fname))
gpt.to_csv(fname, sep="\t")

fname = os.path.join(output_folder, "output_rnn.csv")
print("Saving {}".format(fname))
rnn.to_csv(fname, sep="\t")

