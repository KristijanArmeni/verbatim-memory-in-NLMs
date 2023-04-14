import os
import numpy as np
from scipy.stats import sem, median_absolute_deviation
from matplotlib import pyplot as plt
import pandas as pd
import logging

def filter_and_aggregate(datain, model, model_id, groups, aggregating_metric):
    """
    Parameters:
    ----------
    datain : dataframe
        dataframe which is filtered and aggregated over
    model : string
        string identifying the model in the 'model' column
    model_id: string
        string identifying the id of the model in the 'model_id' column of datain
    groups : list of dicts
        each element is a dict with column name as key and a list of row values as dict value, 
        the first dict should be the variable that's manipulated (e.g. list length),
        the rest are the ones that are held fixed
    """
    # unpack the dictionaries for each variable
    d1, d2, d3, d4 = groups
   
    # define variables for querying below
    var1 = list(d1.keys())[0]  # e.g. .prompt_len
    var2 = list(d2.keys())[0]  # e.g. .list_len
    var3 = list(d3.keys())[0]  # e.g. .context
    var4 = list(d4.keys())[0]  # e.g. .token_positions

    # select the groups based on variable values
    sel = (datain[var1].isin(d1[var1])) & \
          (datain[var2].isin(d2[var2])) & \
          (datain.second_list.isin(["repeat", "permute", "control"])) & \
          (datain.list.isin(["random", "categorized"])) & \
          (datain[var3].isin(d3[var3])) & \
          (datain.model_id==model_id) & \
          (datain.model==model) & \
          (datain.marker.isin([1, 3])) & \
          (datain[var4].isin(d4[var4]))
    
    if sum(sel) == 0:
        logging.info("No rows were selected. Check selection conditions.")
    
    d = datain.loc[sel].copy()
    
    ## Aggregate
    # average separately per list_len, stimulus id (sentid), model (lstm or gpt2), marker (1 or 3), list (random, categorized) and second list (repeated, permuted or control)
    units = [var1, "stimid", "model", "marker", "list", "second_list"]
    
    logging.info("Aggregating metric == {}".format(aggregating_metric))
    logging.info("Aggregating over these variables:")
    display(units)
    
    # aggregate with .groupby and .agg
    dagg = d.groupby(units).agg({"surp": [aggregating_metric, "std"], "token": list}).reset_index()
    dagg.columns = ['_'.join(col_v) if col_v[-1] != '' else col_v[0] for col_v in dagg.columns.values]
    
    target_colname = "surp_" + aggregating_metric    
    ## Compute metric
    dataout = relative_change_wrapper(df_agg=dagg, 
                                      groups = [{"model": [model]}, 
                                                 d1,  # this is the manipulated variable
                                                {"second_list": ["repeat", "permute", "control"]},
                                                {"list": ["categorized", "random"]}
                                                ],
                                      compared_col=target_colname,
                                      )
    
    # rename some column/row names for plotting
    dataout.list = dataout.list.map({"categorized": "semantic", "random": "arbitrary"})
    dataout.rename(columns={"second_list": "condition"}, inplace=True)
    dataout.condition = dataout.condition.map({"control": "Novel", "repeat": "Repeated", "permute": "Permuted"})
    
    return dataout, dagg


def get_relative_change(x1=None, x2=None, labels1=None, labels2=None):
    
    """
    computes relative change across data in x1 and x2. Sizes of arrays x1 and x2
    should match.
    """
    
    # check that any labels match
    if (labels1 is not None) & (labels2 is not None):
        assert (labels1 == labels2).all()
    
    x_del = ((x2-x1)/(x1+x2))
    x_perc = (x2/x1)*100
    return x_del, x_perc

def relative_change_wrapper(df_agg, groups, compared_col):
    
    # unpack the dicts
    g1, g2, g3, g4 = groups
    
    # define coloms for data
    col1 = list(g1.keys())[0]
    col2 = list(g2.keys())[0]
    col3 = list(g3.keys())[0]
    col4 = list(g4.keys())[0]
    
    # apply relative change computation and apply to 
    df_list = []
    for val1 in list(g1.values())[0]:
        for val2 in list(g2.values())[0]:
            for val3 in list(g3.values())[0]:
                for val4 in list(g4.values())[0]:

                    # initialize output dataframe
                    cols = ["x1", "x2", "x_del"]
                    df = pd.DataFrame(columns=cols)
                        
                    # select the correct rows
                    select = (df_agg.loc[:, col1] == val1) & \
                             (df_agg.loc[:, col2] == val2) & \
                             (df_agg.loc[:, col3] == val3) & \
                             (df_agg.loc[:, col4] == val4)
                    
                    tmp = df_agg.loc[select].copy()

                    # get vectors with aggregated surprisal values from first and second list
                    x1=tmp.loc[tmp.marker==1, compared_col].to_numpy()           # average per sentence surprisal on first list
                    x2=tmp.loc[tmp.marker==3, compared_col].to_numpy()           # average per sentence surprisal on second list
                    labels1 = tmp.loc[tmp.marker==1].stimid.to_numpy()  # use sentence id for check
                    labels2 = tmp.loc[tmp.marker==3].stimid.to_numpy()

                    # compute change and populate output dfs
                    x_del, x_perc = get_relative_change(x1=x1, x2=x2, labels1=labels1, labels2=labels2)
                    
                    df["x1"], df["x2"], df["x_del"], df["x_perc"] = x1, x2, x_del, x_perc
                    df[col1], df[col2], df[col3], df[col4] = val1, val2, val3, val4

                    df_list.append(df)
    
    return pd.concat(df_list)

# do layer 11, head index 7
savedir = "/home/ka2773/project/lm-mem/data/fig/"

# load data

dat1 = pd.read_csv("/scratch/ka2773/project/lm-mem/output/ablation/ablate-10-7_merge.csv", sep="\t")
dat0 = pd.read_csv("/scratch/ka2773/project/lm-mem/output/ablation/pretrained_merge.csv", sep="\t")

dat1["model"] = "gpt2"
dat0["model"] = "gpt2"

variables = [{"list_len": [10]},
             {"prompt_len": [8]},
             {"context": ["intact"]},
             {"marker_pos_rel": list(range(1, 10))}]

dat1_, _ = filter_and_aggregate(dat1, model="gpt2", model_id="ablate-10-7", groups=variables, aggregating_metric="mean")
dat0_, _ = filter_and_aggregate(dat0, model="gpt2", model_id="pretrained", groups=variables, aggregating_metric="mean")

y1 = dat1_.x_perc.to_numpy()
y0 = dat0_.x_perc.to_numpy()

fig, ax = plt.subplots(1, 1, figsize=(4, 6))

rel = (dat1_.x2.to_numpy()/dat0_.x2.to_numpy())*100

ax.boxplot(rel, notch=True, bootstrap=2000, labels=['ablation'])
ax.set_xlabel("condition")
ax.set_ylabel("repeat surprisal (%)")
ax.set_title("GPT2")
plt.suptitle("Short-term memory performance after ablation")


fig.savefig(os.path.join(savedir, "test_fig.png"), dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

w = model.transformer.h[4].attn.c_attn.weight.detach()
im = ax.imshow(w[:, 768:768*2], aspect='auto', cmap="RdBu")
ax.set_ylabel('input vector dimension')
ax.set_xlabel('output vector dimension')
ax.set_title('Query weight matrix')

cbar = fig.colorbar(im, ax=ax)

fig.savefig(os.path.join(savedir, "ablation.png"), dpi=300)
