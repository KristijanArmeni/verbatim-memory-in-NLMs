
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
ABLATION_FILENAME_CODES_MULTILAYER = ["01", "23", "45", "67", "89", "1011", "0123", "4567", "891011", "all"]
ABLATION_FILENAME_CODES = ABLATION_FILENAME_CODES_SINGLE_LAYER + ABLATION_FILENAME_CODES_MULTILAYER


def load_ablation_files(datadir: str) -> pd.DataFrame:
    
    dfs = []
    ppls = {}

    for l in ABLATION_FILENAME_CODES:
    
        fn = os.path.join(datadir, f"ablation_gpt2_ablate-{l}-all_sce1_1_3_random_repeat.csv")
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

    fn = os.path.join(datadir, "gpt2_pretrained_sce1_1_3_random_repeat.csv")

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

    tmp = {"model": [], "err": [], "svtype": [], "id": [], "ablation_type": []}

    for i, code in enumerate(ABLATION_FILENAME_CODES + ["unablated", 'rand-init']):
        
        infix =  f"ablate-{code}"
        if code in ["rand-init", "unablated"]:
            infix = code

        fn = os.path.join(datadir, "sva", "zero_attn", f"sva_{infix}_{datasetname}.json")
        logging.info(f"Loading {fn}...")

        with open(fn, "r") as fh:
            sva = json.load(fh)

        ablation_type = "single-layer" if code in ABLATION_FILENAME_CODES_SINGLE_LAYER else "multi-layer"

        tmp['model'].append(f"ablate-{code}")
        tmp['err'].append((1 - sva['accuracy'])*100)
        tmp['svtype'].append(sva['type'])
        tmp['id'].append(sva['id'])
        tmp['ablation_type'].append(ablation_type)

    return pd.DataFrame(tmp)


def plot_sva(ax, svatype:str):
    """
    Plots the results of subject-verb agreement ablation experiments onto two axes.
    """
    
    fname = "C:\\Users\\karmeni1\\project\\lm-mem\\data\\sva\\ablation_sva.csv"
    logging.info(f"Loading {fname}")
    df = pd.read_csv(fname, sep="\t", index_col=0)
    
    selected = (~df.model.isin(["unablated", "rand-init"])) & (df.agr.isin(["SS", "SSS"]))
    data = df.loc[selected].copy()
    data.model = data.model.astype(str).str.replace("ablate-", "")
    df["ablation_type"] = ""
    data.loc[data.model.isin(ABLATION_FILENAME_CODES_SINGLE_LAYER), "ablation_type"] = "single-layer"
    data.loc[data.model.isin(ABLATION_FILENAME_CODES_MULTILAYER), "ablation_type"] = "multi-layer"

    data["err"] = (1- data["acc"])*100

    short_color = "tab:blue"
    long_color = "tab:red"
    short_label = "Short"
    long_label = "Long"
    bar_width = 0.4

    def plot_bars_data(ax, yvals, offset, bar_width, color, label):

        xvals = (np.arange(yvals.shape[-1])+1)+offset    
        ax.barh(y=xvals, width=yvals, height=bar_width, color=color, label=label)

        #yjitter = np.random.uniform(-0.1, 0.1, yvals.shape[0])
        #markers = ['s', 'v']
        #for i in range(yvals.shape[-1]):
        #    ax.scatter(yvals[0:1, i], xvals[i]+yjitter[0], s=10, marker=markers[0], color=color, ec="black", alpha=0.8, zorder=2)
        #    ax.scatter(yvals[1::, i], xvals[i]+yjitter[1], s=10, marker=markers[1], color=color, ec="black", alpha=0.8, zorder=2)

        return ax


    def fetch_yvals(df, iv_vals, iv_col, dv_col):

        vals = []
        for key in iv_vals:
            vals.append(df.loc[df[iv_col] == key, dv_col].item())

        return vals

    data_sel = data.loc[(data.ablation_type == "multi-layer") & (data.deplen == "short")]

    yvals = fetch_yvals(df=data_sel, iv_vals=ABLATION_FILENAME_CODES_MULTILAYER, iv_col="model", dv_col="err")
    yvals = np.stack(yvals)  # shape = (n_conditions, n_ablations)

    plot_bars_data(ax[0], yvals, offset=0.2, bar_width=0.4, color=short_color, label=short_label)

    data_sel = data.loc[(data.ablation_type == "multi-layer") & (data.deplen == "long")]
    yvals = fetch_yvals(df=data_sel, iv_vals=ABLATION_FILENAME_CODES_MULTILAYER, iv_col="model", dv_col="err")
    yvals = np.stack(yvals)

    plot_bars_data(ax[0], yvals, offset=-0.2, bar_width=0.4, color=long_color, label=long_label)   

    # condition 2 barplot
    yvals = data.loc[(data.ablation_type == "single-layer") & (data.deplen == "short")].err.tolist()
    yvals = np.stack(yvals)

    plot_bars_data(ax[1], yvals, offset=0.2, bar_width=0.4, color=short_color, label=short_label)

    yvals = data.loc[(data.ablation_type == "single-layer") & (data.deplen == "long")].err.tolist()
    yvals = np.stack(yvals)
    plot_bars_data(ax[1], yvals, offset=-0.2, bar_width=0.4, color=long_color, label=long_label)

    data = df.loc[df.model.isin(["unablated"]) & (df.deplen == "short") & (df.agr == "SS")].copy()
    unablated1 = (1 - data.acc.item())*100
    
    data = df.loc[df.model.isin(["unablated"]) & (df.deplen == "long") & (df.agr == "SSS")].copy()
    unablated2 = (1 - data.acc.item())*100

    for a in ax:
        l1 = a.axvline(x=50, ymin=0, ymax=a.get_ylim()[-1], linestyle="--", color="tab:gray", zorder=0)
        l2 = a.axvline(x=unablated1, ymin=0, ymax=a.get_ylim()[-1], linestyle=":", linewidth=2, color=short_color)
        l3 = a.axvline(x=unablated2, ymin=0, ymax=a.get_ylim()[-1], linestyle=":", linewidth=2, color=long_color)

    legend1 = plt.legend(handles=[l1, l2, l3], labels=["Chance", "Unablated (short)", "Unablated (long)"], 
                         title="", loc='lower right', bbox_to_anchor=(1.04, 0.03),
                         fontsize=13, title_fontsize=13)
    ax[1].add_artist(legend1)

    ax[1].legend(title="Dependency distance",
                loc="upper right", framealpha=0, facecolor="white", frameon=True,
                fontsize=13, title_fontsize=13)

    # set ticks and axis lable
    ticklabelfs=14
    ax[0].set_yticks(np.arange(10)+1)
    ax[1].set_yticks(np.arange(12)+1)
    ax[0].set_yticklabels(["1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "1-2-3-4", "5-6-7-8", "9-10-11-12", "All"], fontsize=ticklabelfs)
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
        a.vlines(x = y0_m, ymin=0, ymax=len(d)+1, linestyle='--', color='tab:gray', label='Unablated\nmodel', zorder=0)
        a.fill_betweenx(y = np.arange(0, len(d)+2), x1=ci[0], x2=ci[1], alpha=0.2, color='tab:gray', zorder=0)
    
    # ticks and tick labels
    ticklabelfs=14
    ax[0].set_ylim([0, len(y_joint)+1])
    ax[1].set_ylim([0, len(y_ind)+1])

    singlelayer_labels = [i+1 for i in range(12)]
    multilayer_labels = ["1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "1-2-3-4", "5-6-7-8", "9-10-11-12", "All"]
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
                      pd.DataFrame([{"label": "unablated", "median": y0_m, "CI1": ci[0], "CI2": ci[1]}])])

    return ax, data


def ablation_perplexity_plot(ax, ppls: List):


    single_layer_ablations = list(range(12))
    multi_layer_ablations = ['01', '23', "45", "67", '89', '1011', '0123', "4567", '891011', 'all']

    # plot single-layer ablations
    y3 = np.array([ppls[str(k)]['wt103_ppl'] for k in single_layer_ablations])
    y3 = np.log(y3)
    ax[1].barh(np.arange(12)+1, width=y3)
    #ax[1, 1].barh(np.arange(12)+1, width=y3)
    
    # plot multi-layer ablations
    y4 = np.array([ppls[str(k)]['wt103_ppl'] for k in multi_layer_ablations])
    y4 = np.log(y4)

    ax[0].barh(np.arange(len(y4))+1, width=y4)

    #ax[1, 0].set_xlim(0, 150)
    ax[1].set_xticks(np.arange(0, 10, 1))
    ax[1].set_xticklabels([0, "", 2, "", 4, "", 6, "", 8, ""])

    for a in ax:
        a.spines['left'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.tick_params(left=False, labelleft=False)
        
    # plot unablated perplexity as a basline
    ax[0].vlines(x=np.log(ppls["unablated"]["wt103_ppl"]), 
                    ymin=0, 
                    ymax=len(multi_layer_ablations)+1, 
                    linestyle="--", 
                    color='tab:orange', 
                    zorder=2)
    
    ax[1].vlines(x=np.log(ppls["unablated"]["wt103_ppl"]), 
                    ymin=0, ymax=len(single_layer_ablations)+1,
                    linestyle="--",
                    color='tab:orange',
                    zorder=2,
                    label="Unablated\nmodel")

    ticklabelfs=14
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="upper right", fontsize=ticklabelfs)

    ax[1].set_xlabel("Test-set loss", fontsize=ticklabelfs)

    data = pd.DataFrame([{key: str(np.log(ppls[key]['wt103_ppl'])) for key in ppls.keys()}]).T
    data.columns = ["log(ppl)"]


    return ax, data


def annotate_axes(ax):

    arrows_y = -2.6
    texts_y = -2.4

    # left panel
    ax[1, 0].annotate('', xy=(80, arrows_y), xytext=(20, arrows_y),                   
                arrowprops=dict(arrowstyle='<->'),  
                annotation_clip=False)                               

    ax[1, 0].text(x=12, y=texts_y, s='Better memory', fontsize=13, color="tab:gray")
    ax[1, 0].text(x=65, y=texts_y, s='Worse memory', fontsize=13, color="tab:gray")

    # middle panel
    ax[1, 1].annotate('', xy=(5.5, arrows_y), xytext=(1.5, arrows_y),                   
                      arrowprops=dict(arrowstyle='<->'),  
                      annotation_clip=False)

    ax[1, 1].text(x=0, y=texts_y, s='Accurate prediction', fontsize=13, color="tab:gray")
    ax[1, 1].text(x=4, y=texts_y, s='Inaccurate prediction', fontsize=13, color="tab:gray")

    # middle panel
    ax[1, -1].annotate('', xy=(12.5, arrows_y), xytext=(80, arrows_y),                   
                      arrowprops=dict(arrowstyle='<->'),  
                      annotation_clip=False)

    ax[1, -1].text(x=2, y=texts_y, s='Correct agreement', fontsize=13, color="tab:gray")
    ax[1, -1].text(x=60, y=texts_y, s='Incorrect agreement', fontsize=13, color="tab:gray")

    return ax


def generate_plot(datadir: str, timesteps: str):
    """
    The main routinfor drawing the entire plot containing short-term memory, language modeling and subject-verb 
    agreement tasks.
    """
    # ===== LOAD DATA INTO DATAFRAMES ===== #

    dfs, ppls = load_ablation_files(datadir=datadir)
    dfs2, ppls2 = load_intactmodel_files(datadir=datadir)

    #print(dfs)

    # add this to the existing dict
    ppls["unablated"] = ppls2

    # define the timesteps over which the filter_and_aggreagate() should average the surprisals
    if timesteps == "all_tokens":
        selected_timesteps = list([0, 1, 2])
    elif timesteps == "first_token":
        selected_timesteps = list([0])
    elif timesteps == "second_half":
        selected_timesteps = list([4, 5, 6, 7, 8, 9])

    # AVERAGE ACROSS TIME STEPS
    variables = [{"list_len": [3]},
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
    dat0_, _ = filter_and_aggregate(dfs2, 
                                    model="gpt2", 
                                    model_id="pretrained", 
                                    groups=variables, 
                                    aggregating_metric="mean")

    
    # get percent reduction
    y1 = [d.x_perc.to_numpy() for d in dats_]  # median surprisal on second list for each ablation
    y = np.stack(y1)

    y0 = dat0_.x_perc.to_numpy()                 # median surprisal on second list for non-ablated model
    y0_m = np.median(y0)
    bs = bootstrap((y0,), axis=0, statistic=np.median, confidence_level=0.95, n_resamples=10000)

    ci = (bs.confidence_interval.low, bs.confidence_interval.high)

    with plt.style.context('seaborn-ticks'):

        fig, ax = plt.subplots(2, 3, figsize=(14, 9), sharex="col", sharey="row", 
                               gridspec_kw={'height_ratios': [1.9, 2.3], 'width_ratios': [1, 0.9, 1]})

        # plot memory scores
        _, data1 = ablation_memory_plot(ax[:, 0], y, y0_m, ci)

        # plot perplexity values
        _, data2 = ablation_perplexity_plot(ax[:, 1], ppls)

        # plot language-modeling task
        _, _ = plot_sva(ax[:, 2], svatype="congruent")

        for a in ax[1, :]:
            a.tick_params(axis="x", labelsize=13)

        # subtitles
        ax[1, 0].set_xlabel("Surprisal on second list relative to first list (%)", fontsize=13)
        
        if timesteps == "first_token":
            ax[1, 0].set_xlabel("Repeat surprisal on the first token in list (%)", fontsize=13)

        ax = annotate_axes(ax)

        titlesfontsize = 14
        ax[0, 0].set_title("Working memory task", fontsize=titlesfontsize)
        ax[0, 1].set_title("Language modeling (WT-103)", fontsize=titlesfontsize)
        ax[0, 2].set_title("Subject-verb agreement (singular verbs)", fontsize=titlesfontsize)
        fig.supylabel('Ablated layer(s)', fontsize=titlesfontsize)

        plt.suptitle("Effect of attention ablation on memory, language modeling, and subject-verb agreement tasks in GPT-2", fontsize=18)
        
        for a in ax[0, :]:
            a.grid(visible=True, linewidth=0.5)
        for a in ax[1, :]:
            a.grid(visible=True, linewidth=0.5)

        plt.tight_layout()

    return fig, (data1, data2)


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

            #fn = os.path.join(args.savedir, "ablation_sva.csv")
            #logging.info(f"Saving {fn}")
            #d[2].to_csv(fn, sep="\t")

    return 0


if __name__ == "__main__":

    main()