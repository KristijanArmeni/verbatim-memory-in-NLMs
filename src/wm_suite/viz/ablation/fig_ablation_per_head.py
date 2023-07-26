# %%
import os, json, sys

from paths import PATHS

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import median_abs_deviation, bootstrap
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

from src.wm_suite.viz.func import filter_and_aggregate, set_manuscript_style
from src.wm_suite.viz.ablation.fig_topk import plot_box_data
from src.wm_suite.wm_ablation import find_topk_attn
from src.wm_suite.viz.utils import save_png_pdf

import logging
from typing import List, Dict, Tuple
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(message)s")

# %%
def load_ablation_files(datadir: str, layer_head_combinations: List[Tuple]) -> pd.DataFrame:
    
    dfs = []
    ppls = {}

    for tup in layer_head_combinations:

        layer, head = tup
    
        fn = os.path.join(datadir, f"ablation_gpt2_ablate-{layer}-{head}_sce1_1_3_random_repeat.csv")
        fn2 = os.path.join(datadir, f"gpt2_ablate-{layer}-{head}_ppl.json")
        
        logging.info(f"Loading {fn}")

        # add certain columns expected by other functions
        tmp = pd.read_csv(fn, sep="\t")
        tmp['model'] = "gpt2"
        tmp['second_list'] = "repeat"
        tmp['list'] = "random"
        tmp['prompt_len'] = 8
        tmp = tmp.rename(columns={'trialID': 'marker', "stimID": "stimid"})
        tmp['context'] = "intact"
        tmp['model_id'] = f"ablate-{layer}-{head}"
        
        logging.info(f"Loading {fn2}")

        with open(os.path.join(datadir, fn2), "r") as fh:
            ppls[f"{layer}-{head}"] = json.load(fh)
    
        dfs.append(tmp)

    return dfs, ppls

# %%

def plot_bar_data(ax, xloc, y, horizontal, m='o', c=None, a=0.6, label=None):

    if c is None:
        color = "lightsteelblue"
    else:
        color = c


    xjitter = np.random.uniform(-0.12, 0.12, y.size)
    x = xloc + xjitter
    if horizontal:
        y = xloc + xjitter
        x = y

    ym = np.expand_dims(np.median(y), -1)
    bs = bootstrap((y,), axis=0, statistic=np.median, confidence_level=0.95, n_resamples=10000)
    ci = np.expand_dims(np.array((bs.confidence_interval.low, bs.confidence_interval.high)), axis=-1)

    ci[0, :] = ym - ci[0, :]   # subtract lower from median
    ci[1, :] = ci[1, :] - ym   # subtract higher from median

    ax.scatter(x, y, s=10, marker=m, color="silver", alpha=a, zorder=0, label=label)
    ax.bar(x=[xloc], height=ym, yerr=ci, facecolor="none", ecolor="tab:red", edgecolor="black", zorder=1)

    return ax

# %%

datadir = os.path.join(PATHS.data, "ablation", "zero_attn", "per_head")

attn = np.load(os.path.join(PATHS.data, "ablation", "attention_weights_gpt2_colon-colon-p1.npz"))["data"]

matching_dict, _, _ = find_topk_attn(attn, topk=20, tokens_of_interest=[13], seed=12345)
postmatching_dict, _, _ = find_topk_attn(attn, topk=20, tokens_of_interest=[14, 16, 18], seed=12345)
recent_dict, _, _ = find_topk_attn(attn, topk=20, tokens_of_interest=[42, 43, 44], seed=12345)

# %%

def get_combs(lh_dict: Dict) -> List:
    return [(l, h) for l in lh_dict.keys() for h in lh_dict[l]]

def get_repeat_surprisals(df_list):
    # define the timesteps over which the filter_and_aggreagate() should average the surprisals
    selected_timesteps = list([0])

    # grad the first time step (technically, mean computation is applied, but no effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": selected_timesteps}]

    # loop over data and average over time-points
    dats_ = {}
    for l in range(len(df_list)):

        dat_, _ = filter_and_aggregate(df_list[l], 
                                        model="gpt2", 
                                        model_id=df_list[l].model_id.unique().item(), 
                                        groups=variables, 
                                        aggregating_metric="mean")
        
        dats_[df_list[l].model_id.unique().item()] = dat_.x_perc.to_numpy()

    return dats_

# %%

lh_matching = get_combs(matching_dict)
dfs1, ppls1 = load_ablation_files(datadir, layer_head_combinations=lh_matching)

lh_postmatching = get_combs(postmatching_dict)
dfs2, ppls2 = load_ablation_files(datadir, layer_head_combinations=lh_postmatching)

lh_recent = get_combs(recent_dict)
dfs3, ppls3 = load_ablation_files(datadir, layer_head_combinations=lh_recent)

data1 = get_repeat_surprisals(dfs1)
data2 = get_repeat_surprisals(dfs2)
data3 = get_repeat_surprisals(dfs3)


# %%

def make_subplot(ax, data, c):
    offset = 0
    ys = list(data.values())
    labs = [f"L{int(s.split('-')[1])+1}.H{int(s.split('-')[-1])+1}" for s in list(data.keys())]


    for i, k in enumerate(range(1, len(data)+1)):
        #
        plot_bar_data(ax, k-offset, ys[i], horizontal=False, c="silver", a=0.4,)

        ax.set_ylim([0, 100])
        ax.set_xticks(list(range(1, len(labs)+1)))
        ax.set_xticklabels(labs, rotation=90, fontsize=13)

    return ax

clrs = list(sns.color_palette("deep").as_hex())

fig, axs = plt.subplots(1, 3, figsize=(14, 3.5), sharey="all")

ax1 = make_subplot(axs[0], data1, c=clrs[2])
ax2 = make_subplot(axs[1], data2, c=clrs[0])
ax3 = make_subplot(axs[2], data3, c=clrs[1])

ax1.set_title("Matching heads", fontsize=14, fontweight="semibold", c="#2ca02c")
ax2.set_title("Post-match heads", fontsize=14, fontweight="semibold", c="#1f77b4")
ax3.set_title("Recent-tokens heads", fontsize=14, fontweight="semibold", c="#ff7f0e")

ax1.set_ylabel("Repeat surprisal (%)", fontsize=13)
ax1.set_yticks([0, 20, 40, 60, 80, 100])
ax1.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=13)

fig.supxlabel("Ablated head (layer and head index)", fontsize=13)

for a in axs:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.yaxis.set_minor_locator(AutoMinorLocator(2))
    a.grid(visible=True, linewidth=0.5, which="both")

plt.tight_layout()
plt.show()

savename = os.path.join(PATHS.root, "fig", "ablation", "topk", "topk_single_head")
save_png_pdf(fig, savename)

# %%
def generate_plot():


    # load memory data

    d = "C:\\users\\karmeni1\\project\\lm-mem\\data\\ablation\\zero_attn\\per_head\\"

    ablated_layers = list(range(12))
    ablated_heads = list(range(12))
    combinations = list(product(ablated_layers, ablated_heads))
    dfs, ppls = load_ablation_files(d, layer_head_combinations=combinations)

    # define the timesteps over which the filter_and_aggreagate() should average the surprisals
    selected_timesteps = list([0])

    # grad the first time step (technically, mean computation is applied, but no effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": selected_timesteps}]

    # loop over data and average over time-points
    dats_ = {}
    for l in range(len(dfs)):

        dat_, _ = filter_and_aggregate(dfs[l], 
                                       model="gpt2", 
                                       model_id=dfs[l].model_id.unique().item(), 
                                       groups=variables, 
                                       aggregating_metric="mean")
        
        dats_[dfs[l].model_id.unique().item()] = np.median(dat_.x_perc.to_numpy())

    # load sva
    fn = "C:\\users\\karmeni1\\project\\lm-mem\\data\\sva\\per_head\\ablation_sva_per_head.csv"
    dat = pd.read_csv(fn, sep="\t")

    ablated_layers = list(range(12))
    ablated_heads = list(range(12))

    combinations = list(product(ablated_layers, ablated_heads))

    y1 = np.zeros(shape=(12, 12))
    y2 = np.zeros(shape=(12, 12))
    y3 = np.zeros(shape=(12, 12))
    for tup in combinations:
        l, h = tup
        y1[h, l] = dats_[f"ablate-{l}-{h}"]
        y2[h, l] = np.log(ppls[f"{l}-{h}"]["wt103_ppl"])
        y3[h, l] = (1 - dat.loc[dat.model == f"ablate-{l}-{h}", "acc"].item())*100

    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3), sharex="all", sharey="all")

    cmap = plt.cm.Blues

    im1 = ax[0].imshow(y1, vmin=0, cmap=cmap)
    im2 = ax[1].imshow(y2, cmap=cmap)
    im3 = ax[2].imshow(y3, vmin=0, cmap=cmap)

    cax1 = ax[0].inset_axes([1.04, 0, 0.05, 1])
    cax2 = ax[1].inset_axes([1.04, 0, 0.05, 1])
    cax3 = ax[2].inset_axes([1.04, 0, 0.05, 1])

    cbar1 = plt.colorbar(im1, ax=ax[0], cax=cax1)
    cbar2 = plt.colorbar(im2, ax=ax[1], cax=cax2)
    cbar3 = plt.colorbar(im3, ax=ax[2], cax=cax3)
    
    cbar1.ax.set_ylabel("Repeat surprisal (%)")
    cbar2.ax.set_ylabel("Test-set loss")
    cbar3.ax.set_ylabel("Error rate (%)")

    ax[0].set_yticks(np.arange(0, 12, 2))
    ax[0].set_yticklabels(np.arange(0, 12, 2)+1)
    for a in ax:
        a.set_xticks(np.arange(0, 12, 2))
        a.set_xticklabels(np.arange(0, 12, 2)+1)

    ax[0].set_ylabel("Head", fontsize=12)
    ax[1].set_xlabel("Layer", fontsize=12)

    ax[0].set_title("Short-term memory")
    ax[1].set_title("Language modeling")
    ax[2].set_title("Subject-verb agreement")

    fig.suptitle("Ablation of single attention heads in GPT-2", fontsize=14)
    plt.tight_layout()
    plt.show()

    return 0


# %%
def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir")

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    datadir = "C:\\users\\karmeni1\\project\\lm-mem\\data\\sva\\per_head"
    fig = generate_plot(datadir=datadir)

# %%
if __name__ == "__main__":

    main()