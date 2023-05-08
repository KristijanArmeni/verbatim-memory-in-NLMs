
import os, json, sys

sys.path.append(os.environ['PROJ_ROOT'])
sys.path.append("/home/ka2773/project/lm-mem/src")

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import median_abs_deviation, bootstrap
from matplotlib import pyplot as plt
import seaborn as sns
from src.wm_suite.viz.func import filter_and_aggregate, set_manuscript_style
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")

ABLATION_FILENAME_CODES_SINGLE_LAYER = [str(i) for i in range(12)]
ABLATION_FILENAME_CODES_MULTILAYER = ["01", "23", "0123", "89", "1011", "891011", "all"]
ABLATION_FILENAME_CODES = ABLATION_FILENAME_CODES_SINGLE_LAYER + ABLATION_FILENAME_CODES_MULTILAYER


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


def load_sva_files(datadir:str, datasetname:str):

    tmp = {"model": [], "acc": [], "svtype": [], "id": [], "ablation_type": []}

    for i, code in enumerate(ABLATION_FILENAME_CODES + ["unablated"]):
        
        infix =  f"ablate-{code}"
        if code in ["rand-init", "unablated"]:
            infix = code

        fn = os.path.join(datadir, "sva", f"sva_{infix}_{datasetname}.json")
        logging.info(f"Loading {fn}...")

        with open(fn, "r") as fh:
            sva = json.load(fh)

        ablation_type = "single-layer" if code in ABLATION_FILENAME_CODES_SINGLE_LAYER else "multi-layer"

        tmp['model'].append(f"ablate-{code}")
        tmp['acc'].append((1 - sva['accuracy'])*100)
        tmp['svtype'].append(sva['type'])
        tmp['id'].append(sva['id'])
        tmp['ablation_type'].append(ablation_type)

    return pd.DataFrame(tmp)


def plot_sva(ax):
    """
    Plots the results of subject-verb agreement ablation experiments onto two axes.
    """
    
    d1 = load_sva_files(datadir='C:\\Users\\karmeni1\\project\\lm-mem\\data\\', datasetname="linzen2016")
    d2 = load_sva_files(datadir='C:\\Users\\karmeni1\\project\\lm-mem\\data\\', datasetname="lakretz2021_A")
    d1['ds'] = 'Linzen 2016 (corpus)'
    d2['ds'] = 'Lakretz 2021 (synthetic)'

    # this will be saved along with the figure
    data = pd.concat([d1, d2])

    # store unablated error-rates
    linzen_unablated = d1.loc[(d1.model == "ablate-unablated"), "acc"].item()
    lakretz_unablated = d2.loc[(d2.model == "ablate-unablated"), "acc"].item()

    d1 = d1.loc[~d1.model.isin(['ablate-unablated']), :]
    d2 = d2.loc[~d2.model.isin(['ablate-unablated']), :]

    d1_color = "tab:blue"
    d2_color = "tab:red"
    d1_label = "Linzen (2016)\n(corpus)"
    d2_label = "Lakretz (2021)\n(synthetic)"
    bar_width = 0.4

    yvals = d1.loc[d1.ablation_type == "multi-layer", "acc"].tolist()
    xvals = (np.arange(len(yvals))+1)+0.2    
    ax[0].barh(y=xvals, width=yvals, height=bar_width, color=d1_color, label=d1_label)

    yvals = d2.loc[d2.ablation_type == "multi-layer", "acc"].tolist()
    xvals = (np.arange(len(yvals))+1)-0.2    
    ax[0].barh(y=xvals, width=yvals, height=bar_width, color=d2_color, label=d2_label)


    yvals = d1.loc[d1.ablation_type == "single-layer", "acc"].tolist()
    xvals = (np.arange(len(yvals))+1)+0.2
    ax[1].barh(y=xvals, width=yvals, height=bar_width, color=d1_color, label=d1_label)
    
    yvals = d2.loc[d2.ablation_type == "single-layer", "acc"].tolist()
    xvals = (np.arange(len(yvals))+1)-0.2
    ax[1].barh(y=xvals, width=yvals, height=bar_width, color=d2_color, label=d2_label)


    for a in ax:
        a.axvline(x=50, ymin=0, ymax=a.get_ylim()[-1], linestyle="--", color="tab:gray", zorder=0)
        a.axvline(x=linzen_unablated, ymin=0, ymax=a.get_ylim()[-1], linestyle=":", linewidth=2, color=d1_color)
        a.axvline(x=lakretz_unablated, ymin=0, ymax=a.get_ylim()[-1], linestyle=":", linewidth=2, color=d2_color)

    ax[1].text(50.5, 1, "Chance", rotation=-90, fontsize=13)
    ax[1].annotate(xy=(linzen_unablated, 12.5), xytext=(35, 12.5), text="Unablated", fontsize=13,
                   arrowprops=dict(facecolor='black', arrowstyle="-|>"))

    ax[1].legend(title="Dataset", 
                loc="lower right", 
                bbox_to_anchor=(1.07, 0.34), framealpha=0, facecolor="white", frameon=True,
                fontsize=11, title_fontsize=11)

    # set ticks and axis lable
    ticklabelfs=14
    ax[0].set_yticks(np.arange(7)+1)
    ax[1].set_yticks(np.arange(12)+1)
    ax[0].set_yticklabels(["1-2", "3-4", "1-2-3-4", "9-10", "11-12", "9-10-11-12", "All"], fontsize=ticklabelfs)
    ax[1].set_yticklabels([i+1 for i in range(12)], fontsize=ticklabelfs)

    ax[1].set_xlabel("Error rate (%)", fontsize=ticklabelfs)

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    return ax, data


def ablation_memory_plot(ax, y, y0_m, ci):
    
    n_layers = 12
    y_ind = y[0:n_layers]
    y_joint = y[n_layers::] 

    markersize=9
    
    # plot boxplots and jitter datapoints
    ax1_data = ax[1].boxplot(y_ind.T, notch=True, vert=False, showfliers=False, bootstrap=10000)
    for i in range(y_ind.shape[-1]):
        yjitter = np.random.uniform(-0.18, 0.18, len(y_ind))
        ax[1].scatter(y_ind[:, i], (np.arange(len(y_ind))+1)+yjitter, s=markersize, color="lightsteelblue", alpha=0.5, zorder=0)

    ax0_data = ax[0].boxplot(y_joint.T, notch=True, vert=False, showfliers=False, bootstrap=10000)
    for i in range(y_joint.shape[-1]):
        yjitter = np.random.uniform(-0.2, 0.2, len(y_joint))
        ax[0].scatter(y_joint[:, i], (np.arange(len(y_joint))+1)+yjitter, s=markersize, color="lightsteelblue", alpha=0.5, zorder=0)
    
    # plot shaded vertical line for the unablated model
    for a, d in zip(ax[:], (y_joint, y_ind)):
        a.vlines(x = y0_m, ymin=0, ymax=len(d)+1, linestyle='--', color='tab:gray', label='Unablated model', zorder=0)
        a.fill_betweenx(y = np.arange(0, len(d)+2), x1=ci[0], x2=ci[1], alpha=0.2, color='tab:gray', zorder=0)
    
    # ticks and tick labels
    ticklabelfs=14
    ax[0].set_ylim([0, len(y_joint)+1])
    ax[1].set_ylim([0, len(y_ind)+1])

    singlelayer_labels = [i+1 for i in range(12)]
    multilayer_labels = ["1-2", "3-4", "1-2-3-4", "9-10", "11-12", "9-10-11-12", "All"]
    ax[0].set_yticklabels(multilayer_labels, fontsize=ticklabelfs)
    ax[1].set_yticklabels(singlelayer_labels, fontsize=ticklabelfs)

    # spines
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    ax[1].legend(loc="lower right", fontsize=ticklabelfs)

    # get data for reporting numerical values on boxplots
    data1 = [item.get_xdata() for item in ax1_data['boxes']]
    data0 = [item.get_xdata() for item in ax0_data['boxes']]
    lower_pannel_list = [{"label": singlelayer_labels[i], "median": d[3], "CI1": d[2], "CI2": d[4]}  
                         for i, d in enumerate(data1)]
    upper_pannel_list = [{"label": multilayer_labels[i], "median": d[3], "CI1": d[2], "CI2": d[4]}  
                         for i, d in enumerate(data0)]

    data = pd.concat([pd.DataFrame(lower_pannel_list), 
                      pd.DataFrame(upper_pannel_list),
                      pd.DataFrame([{"label": "unablated", "median": y0_m, "CI1": ci[0], "CI2": ci[0]}])])
    
    print(data)
    

    return ax, data


def ablation_perplexity_plot(ax, ppls: List):


    single_layer_ablations = list(range(12))
    multi_layer_ablations = ['01', '23', '0123', '89', '1011', '891011', 'all']

    # plot perplexities
    y3 = np.array([ppls[str(k)]['wt103_ppl'] for k in single_layer_ablations])
    ax[1, 0].barh(np.arange(12)+1, width=y3)
    
    # plot perplexities as barplots
    y4 = np.array([ppls[str(k)]['wt103_ppl'] for k in multi_layer_ablations])

    ax[0, 0].barh(np.arange(7)+1, width=y4)
    ax[0, 1].barh(np.arange(7)+1, width=y4)
    ax[0, 0].set_xlim(0, 150)
    ax[0, 0].set_xticks([0, 50, 100, 150])
    ax[0, 0].set_xticklabels([0, 50, 100, 150])
    ax[0, 1].set_xlim(350, None)
    ax[0, 1].set_xticks([350, 500, 1000, 1500])
    ax[0, 1].set_xticklabels(["", 500, "", 1500])

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
                    label="Unablated\nmodel")
        
    # add the // mark
    for a in ax[:, 0]:
        a.text(a.get_xlim()[-1]+20, -0.3, "//", fontsize=13)

        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    ticklabelfs=14
    handles, labels = ax[1, 0].get_legend_handles_labels()
    ax[1, 1].legend(handles, labels, loc="lower right", fontsize=ticklabelfs-1)

    data = pd.DataFrame([{key: str(ppls[key]['wt103_ppl']) for key in ppls.keys()}]).T
    data.columns = ["ppl"]
    print(data)

    return ax, data


def generate_plot(datadir: str, timesteps: str):
    """
    The main routinfor drawing the entire plot containing short-term memory, language modeling and subject-verb 
    agreement tasks.
    """
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
    y = np.stack(y1)


    y0 = dat0_.x_perc.to_numpy()                 # median surprisal on second list for non-ablated model
    y0_m = np.median(y0)
    print(y0.shape)
    bs = bootstrap((y0,), axis=0, statistic=np.median, confidence_level=0.95, n_resamples=10000)

    ci = (bs.confidence_interval.low, bs.confidence_interval.high)
    print(y0_m, ci)

    with plt.style.context('seaborn-ticks'):

        fig, ax = plt.subplots(2, 4, figsize=(14, 8), sharex="col", sharey="row", 
                               gridspec_kw={'height_ratios': [1.4, 2.5], 'width_ratios': [1.3, 0.4, 0.4, 0.8]})

        # plot memory scores
        _, data1 = ablation_memory_plot(ax[:, 0], y, y0_m, ci)

        # plot perplexity values
        _, data2 = ablation_perplexity_plot(ax[:, 1:4], ppls)

        # plot language-modeling task
        _, data3 = plot_sva(ax[:, 3])

        for a in ax[1, :]:
            a.tick_params(axis="x", labelsize=13)

        # subtitles
        ax[1, 0].set_xlabel("Surprisal on second list relative to first list (%)", fontsize=13)
        
        if timesteps == "first_token":
            ax[1, 0].set_xlabel("Repeat surprisal on the first token in list (%)", fontsize=13)

        ax[1, 0].annotate('', xy=(90, -3), xytext=(15,-3),                   
                    arrowprops=dict(arrowstyle='<->'),  
                    annotation_clip=False)                               

        ax[1, 0].text(x=19, y=-2.8, s='Better memory', fontsize=13)
        ax[1, 0].text(x=58, y=-2.8, s='Worse memory', fontsize=13)

        titlesfontsize = 14
        ax[0, 0].set_title("Working memory task", fontsize=titlesfontsize)
        ax[0, 3].set_title("Subject-verb agreement", fontsize=titlesfontsize)
        fig.text(0.61, 0.06, 'Test-set perplexity', ha='center', fontsize=titlesfontsize)
        fig.text(0.61, 0.92, 'Language modeling (Wikitext-103)', ha='center', fontsize=titlesfontsize)
        fig.supylabel('Ablated layer(s)', fontsize=titlesfontsize)

        plt.suptitle("Effect of GPT-2 attention ablation on memory, language modeling, and subject-verb agreement tasks", fontsize=18)
        
        for a in ax[0, :]:
            a.grid(visible=True, linewidth=0.5)
        for a in ax[1, :]:
            a.grid(visible=True, linewidth=0.5)

        plt.tight_layout()

    return fig, (data1, data2, data3)


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
    with plt.style.context('seaborn-ticks'):
    
        #fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex="all")

        #plot_sva(ax=ax)

        #plt.show()

        fig, d = generate_plot(datadir=args.datadir, timesteps="first_token")
        plt.show()

        if args.savedir:
            fn = os.path.join(args.savedir, "ablation_single-multi_first-token.png")
            print(f"Saving {fn}")
            fig.savefig(fn, dpi=300)

            fn = os.path.join(args.savedir, "ablation_single-multi_first-token.pdf")
            print(f"Saving {fn}")
            fig.savefig(fn, transparent=True, bbox_inches="tight")

            fn = os.path.join(args.savedir, "ablation_memory.csv")
            logging.info(f"Saving {fn}")
            d[0].to_csv(fn, sep="\t")

            fn = os.path.join(args.savedir, "ablation_lm.csv")
            logging.info(f"Saving {fn}")
            d[1].to_csv(fn, sep="\t")

            fn = os.path.join(args.savedir, "ablation_sva.csv")
            logging.info(f"Saving {fn}")
            d[2].to_csv(fn, sep="\t")

    return 0


if __name__ == "__main__":

    main()