
import os, json, sys

sys.path.append(os.environ['PROJ_ROOT'])
sys.path.append("/home/ka2773/project/lm-mem/src")

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import median_abs_deviation
from matplotlib import pyplot as plt
from src.wm_suite.viz.func import filter_and_aggregate, set_manuscript_style
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")

ABLATION_FILENAME_CODES = [str(i) for i in range(12)] + ["01", "23", "56", "711", "0123", "56711", "all"]


def load_ablation_files(datadir: str) -> pd.DataFrame:
    
    dfs = []
    ppls = {}

    for l in ABLATION_FILENAME_CODES:
    
        fn = os.path.join(datadir, f"ablation_gpt2_ablate-{l}-all_sce1_1_10_random_repeat.csv")
        fn2 = os.path.join(datadir, f"gpt2_ablate-{l}-all_ppl.json")
        
        logging.info(f"Loading {fn}")

        tmp = pd.read_csv(fn, sep="\t")
        tmp['model'] = "gpt2"
        tmp['second_list'] = "repeat"
        tmp['list'] = "random"
        tmp['prompt_len'] = 8
        tmp = tmp.rename(columns={'trialID': 'marker', "stimID": "stimid"})
        tmp['context'] = "intact"
        tmp['model_id'] = f"ablate-{l}-all"
        
        logging.info(f"Loading {fn2}")

        with open(os.path.join(datadir, fn2), "r") as fh:
            ppls[l] = json.load(fh)
    
        dfs.append(tmp)

    return dfs, ppls


def load_intactmodel_files(datadir: str) -> pd.DataFrame:

    fn = os.path.join(datadir, "gpt2_pretrained_sce1_1_10_random_repeat.csv")

    logging.info(f"Loading {fn}")

    df = pd.read_csv(fn, sep="\t")
    df["model"] = "gpt2"
    df["model_id"] = "pretrained"
    df["context"] = "intact"
    df['prompt_len'] = 8
    df = df.rename(columns={'trialID': 'marker', "stimID": "stimid"})

    # load perplexity for unablated model
    fn2 = os.path.join(datadir, "gpt2-small_ppl.json")
    with open(fn2, "r") as fh:
        ppl = json.load(fh)

    return df, ppl


def ablation_memory_plot(ax, y, y0_m, y0_sd):
    
    n_layers = 12
    y_ind = y[0:n_layers]
    y_joint = y[n_layers::] 
    
    markersize=9
    
    # plot boxplots and jitter datapoints
    ax[1].boxplot(y_ind.T, notch=True, vert=False, showfliers=False, bootstrap=2000)
    for i in range(y_ind.shape[-1]):
        yjitter = np.random.uniform(-0.18, 0.18, len(y_ind))
        ax[1].scatter(y_ind[:, i], (np.arange(len(y_ind))+1)+yjitter, s=markersize, color="lightsteelblue", alpha=0.5, zorder=0)

    ax[0].boxplot(y_joint.T, notch=True, vert=False, showfliers=False, bootstrap=2000)
    for i in range(y_joint.shape[-1]):
        yjitter = np.random.uniform(-0.2, 0.2, len(y_joint))
        ax[0].scatter(y_joint[:, i], (np.arange(len(y_joint))+1)+yjitter, s=markersize, color="lightsteelblue", alpha=0.5, zorder=0)
    
    # plot shaded vertical line for the unablated model
    for a, d in zip(ax[:], (y_joint, y_ind)):
        a.vlines(x = y0_m, ymin=0, ymax=len(d)+1, linestyle='--', color='tab:gray', label='unablated model', zorder=0)
        a.fill_betweenx(y = np.arange(0, len(d)+2), x1=y0_m-y0_sd, x2=y0_m+y0_sd, alpha=0.2, color='tab:gray', zorder=0)
    
    # ticks and tick labels
    ticklabelfs=13
    ax[0].set_ylim([0, len(y_joint)+1])
    ax[1].set_ylim([0, len(y_ind)+1])
    ax[0].set_yticklabels(["1-2", "3-4", "6-7", "8-12", "1-2-3-4", "6-7-8-12", "all"], fontsize=ticklabelfs)
    ax[1].set_yticklabels([i+1 for i in range(12)], fontsize=ticklabelfs)

    # spines
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    ax[1].legend(loc="lower right", fontsize=ticklabelfs)
    
    return ax


def ablation_perplexity_plot(ax, ppls: List):

    single_layer_ablations = list(range(12))
    multi_layer_ablations = ['01', '23', '56', '711', '0123', '56711', 'all']

    # plot perplexities
    y3 = np.array([ppls[str(k)]['wt103_ppl'] for k in single_layer_ablations])
    ax[1, 0].barh(np.arange(12)+1, width=y3)
    
    # plot perplexities as barplots
    y4 = np.array([ppls[str(k)]['wt103_ppl'] for k in multi_layer_ablations])

    ax[0, 0].barh(np.arange(7)+1, width=y4)
    ax[0, 1].barh(np.arange(7)+1, width=y4)
    ax[0, 0].set_xlim(0, 120)
    ax[0, 1].set_xlim(300, None)
    
    for a in ax[:, 1]:
        a.spines['left'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.tick_params(left=False, labelleft=False)
        
    # plot unablated perplexity as a basline
    ax[0, 0].vlines(x=ppls["unablated"]["wt103_ppl"], 
                    ymin=0, 
                    ymax=len(multi_layer_ablations)+1, 
                    linestyle="--", 
                    color='tab:orange', 
                    zorder=2)
    
    ax[1, 0].vlines(x=ppls["unablated"]["wt103_ppl"], 
                    ymin=0, ymax=len(single_layer_ablations)+1,
                    linestyle="--",
                    color='tab:orange',
                    zorder=2,
                    label="unablated\nmodel")
        
    # add the // mark
    for a in ax[:, 0]:
        a.text(a.get_xlim()[-1]+20, -0.25, "//", fontsize=13)

        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    ticklabelfs=13
    handles, labels = ax[1, 0].get_legend_handles_labels()
    ax[1, 1].legend(handles, labels, loc="lower right", fontsize=ticklabelfs-1)

    return ax


def generate_plot(datadir: str, timesteps: str):

    # ===== LOAD DATA INTO DATAFRAMES ===== #

    dfs, ppls = load_ablation_files(datadir=datadir)
    dfs2, ppls2 = load_intactmodel_files(datadir=datadir)

    # add this to the existing dict
    ppls["unablated"] = ppls2

    # define the timesteps over which the filter_and_aggreagate() should average the surprisals
    if timesteps == "all_tokens":
        selected_timesteps = list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    elif timesteps == "first_token":
        selected_timesteps = list([0])
    elif timesteps == "second_half":
        selected_timesteps = list([4, 5, 6, 7, 8, 9])


    # AVERAGE ACROSS TIME STEPS
    variables = [{"list_len": [10]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": selected_timesteps}]

    # loop over data and average over time-points
    dats_ = []
    for l in range(len(dfs)):

        dat_, _ = filter_and_aggregate(dfs[l], 
                                       model="gpt2", 
                                       model_id=dfs[l].model_id.unique().item(), 
                                       groups=variables, 
                                       aggregating_metric="mean")
        
        dats_.append(dat_)

    # take the mean over timesteps for the unablated model
    dat0_, _ = filter_and_aggregate(dfs2, model="gpt2", model_id="pretrained", groups=variables, aggregating_metric="mean")

    # get percent reduction
    y1 = [d.x_perc.to_numpy() for d in dats_]  # median surprisal on second list for each ablation

    y0 = dat0_.x_perc.to_numpy()                 # median surprisal on second list for non-ablated model
    y0_m = np.median(y0)
    y0_sd = median_abs_deviation(y0)

    # compute ablation surprisals relative to non-ablated
    y = np.stack(y1)


    with plt.style.context('seaborn-whitegrid'):

        fig, ax = plt.subplots(2, 3, figsize=(11, 9), sharex="col", sharey='row', 
                               gridspec_kw={'height_ratios': [1.3, 2.7], 'width_ratios': [2, 0.5, 0.5]})

        _ = ablation_memory_plot(ax[:, 0], y, y0_m, y0_sd)
        _ = ablation_perplexity_plot(ax[:, 1::], ppls)

        for a in ax[1, :]:
            a.tick_params(axis="x", labelsize=13)

        # subtitles
        ax[1, 0].set_xlabel("Surprisal on second list relative to first list (%)", fontsize=13)
        
        if timesteps == "first_token":
            ax[1, 0].set_xlabel("Surprisal on first token of second list relative to first token on first list (%)", fontsize=13)

        ax[1, 0].annotate('', xy=(80, -1.8), xytext=(20,-1.8),                   
                    arrowprops=dict(arrowstyle='<->'),  
                    annotation_clip=False)                               

        ax[1, 0].annotate('better memory', xy=(22,-1.7), xytext=(22,-1.7), annotation_clip=False)
        ax[1, 0].annotate('worse memory', xy=(61,-1.7), xytext=(61,-1.7), annotation_clip=False)

        ax[0, 0].set_title("Working memory task", fontsize=13)

        fig.text(0.84, 0.045, 'Test set perplexity', ha='center', fontsize=13)
        fig.text(0.84, 0.926, 'Language modeling (Wikitext-103)', ha='center', fontsize=13)
        fig.text(0.02, 0.5, 'ablated layer(s)', ha='center', rotation='vertical', fontsize=14)

        plt.suptitle("Effect of GPT-2 attention ablation on memory and language modeling tasks", fontsize=15)
        plt.tight_layout()

    return fig


def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    set_manuscript_style()

    # actual script
    with plt.style.context('seaborn-whitegrid'):
    
        fig = generate_plot(datadir=args.datadir, timesteps="first_token")
        plt.show()

        if args.savedir:
            fn = os.path.join(args.savedir, "ablation_single-multi_first-token.png")
            print(f"Saving {fn}")
            fig.savefig(fn, dpi=300)

            fn = os.path.join(args.savedir, "ablation_single-multi_first-token.pdf")
            print(f"Saving {fn}")
            fig.savefig(fn, transparent=True, bbox_inches="tight")

    return 0


if __name__ == "__main__":

    main()