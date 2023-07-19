
# run `python proj_paths.py in the project roots to set paths
#%%
# utilities
import os, json
from itertools import product
import logging
from typing import List, Dict, Tuple

# main modules
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

# own modules
from src.wm_suite.viz.func import filter_and_aggregate, set_manuscript_style
from src.wm_suite.viz.utils import save_png_pdf, clrs
from src.wm_suite.wm_ablation import find_topk_attn, from_dict_to_labels, from_labels_to_dict

logging.basicConfig(level=logging.INFO, format="%(message)s")


# %%
def construct_filnames():

    csvs = {}
    jsons = {}

    topk = [5, 10, 15, 20]
    head_labels = ["copying", "previous", "matching", "copying-and-matching"]
    control_labels = [f"copying-ctrl{i}" for i in range(5)] + [f"previous-ctrl{i}" for i in range(5)] + [f"matching-ctrl{i}" for i in range(5)] + [f"copying-and-matching-ctrl{i}" for i in range(5)]
    head_types = head_labels + control_labels

    combinations = list(product(topk, head_types))

    for tup in combinations:

        k, head_type = tup
        csvs[f"ablate-top{k}-{head_type}"] = f"ablation_gpt2_ablate-top{k}-{head_type}_sce1_1_3_random_repeat.csv"
        jsons[f"ablate-top{k}-{head_type}"] = f"gpt2_ablate-top{k}-{head_type}_ppl.json"

    csvs["matching-bottom5"] = "ablation_gpt2_ablate-matching-bottom5_sce1_1_3_random_repeat.csv"
    jsons["matching-bottom5"] = "gpt2_ablate-matching-bottom5_ppl.json"

    csvs["induction-matching-intersect"] = "ablation_gpt2_ablate-induction-matching-intersect_sce1_1_3_random_repeat.csv"
    jsons["induction-matching-intersect"] = "gpt2_ablate-induction-matching-intersect_ppl.json"

    csvs["unablated"] = "gpt2_pretrained_sce1_1_3_random_repeat.csv"
    jsons["unablated"] = "gpt2-small_ppl.json"
    
    return csvs, jsons


# %%
def read_csv_add_columns(fullfile:str, model_id_string: str) -> pd.DataFrame:
    """
    Read a .csv file into a dataframe and add a few columns such that it can work with 
    viz.func.filter_and_aggregate
    """
    tmp = pd.read_csv(fullfile, sep="\t")

    tmp['model'] = "gpt2"
    tmp['second_list'] = "repeat"
    tmp['list'] = "random"
    tmp['prompt_len'] = 8
    tmp = tmp.rename(columns={'trialID': 'marker', "stimID": "stimid"})
    tmp['context'] = "intact"
    tmp['model_id'] = model_id_string

    return tmp


# %%
def load_memory_ppl_files(datadir: str, csvfiles: Dict, jsonfiles: Dict) -> List[pd.DataFrame]:

    dfs = []

    for csvkey, jsonkey in zip(csvfiles.keys(), jsonfiles.keys()):

        csv_file = csvfiles[csvkey]
        json_file = jsonfiles[jsonkey]

        fn = os.path.join(datadir, csv_file)
        
        logging.info(f"Loading {fn}")
        dftmp = read_csv_add_columns(fullfile=fn, model_id_string=csvkey)
        dfs.append(dftmp)
        
        fn2 = os.path.join(datadir, json_file)
        logging.info(f"Loading {fn2}")

        with open(os.path.join(datadir, fn2), "r") as fh:
            dftmp["wt103_ppl"] = json.load(fh)["wt103_ppl"]

    return dfs


# %%
def load_sva_files(datadir: str):

    return pd.read_csv(os.path.join(datadir, "ablate_top-k_sva.csv"), sep="\t")


# %%
def compute_repeat_surprisals(dataframes: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A wrapper around filter_and_aggregate(). Loops over data frames in <dataframes> and
    and computes repeat surprisals on them. It also extracts wt103 perplexity from them.
    """

    # AVERAGE ACROSS TIME STEPS
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [0]}]

    # loop over data and average over time-points
    dats_ = []
    labs = []
    wt103_ppls = []
    for l in range(len(dataframes)):

        dat_, _ = filter_and_aggregate(dataframes[l], 
                                    model="gpt2", 
                                    model_id=dataframes[l].model_id.unique().item(), 
                                    groups=variables, 
                                    aggregating_metric="mean")
        
        dats_.append(dat_)
        wt103_ppls.append(dataframes[l].wt103_ppl.unique().item())
        labs.append(dataframes[l].model_id.unique().item())

    xlabels = np.array(labs)
    y = np.stack([d.x_perc.to_numpy() for d in dats_])
    y_ppl = np.log(np.stack(wt103_ppls))

    return y, y_ppl, xlabels

# %%
def get_sva_arrays_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

    df.model = df.model.str.replace("top-", "top")
    
    df["err"] = ((1 - df.acc)*100).round(2)
    
    # extract the right columns
    y_sva = df.err.to_numpy()
    sva_labels = df.model.to_numpy()

    return y_sva, sva_labels


# %% get percent reduction
def select_conditions(labels: np.ndarray, substring: str, match:str="substring") -> np.ndarray:
    
    if match == "substring":
        ids = np.stack([i for i, l in enumerate(labels) if substring in l])

    elif match == "exact":
        ids = np.stack([i for i, l in enumerate(labels) if substring == l])

    print(f"Selected {len(ids)} labels: {labels[ids]}")

    return ids


# %%
def aggregate_over(array: np.ndarray, labels: np.ndarray, substring: str, match: str, agg_func):
    """
    Finds indices in `array` by matching `substring` in `labels` and applies `agg_func`
    across the first dimenion of `array`: `agg_func(array[sel_indx, ...])`
    """

    inds = select_conditions(labels, substring, match)

    array_agg = agg_func(array[inds, ...], axis=0)

    return array_agg        


# %%
def prepare_data(d_mem, d_ppl, d_sva, labels, target_label, labels_sva, target_label_sva):

    ids = []
    ids_sva = []
    for k in [5, 10, 15, 20]:
        ids.append(select_conditions(labels, f"ablate-top{k}-{target_label}", match="exact").tolist()[0])
        ids_sva.append(select_conditions(labels_sva, f"top{k}-{target_label_sva}", match="exact").tolist()[0])

    yA = d_mem[ids]
    yB = d_ppl[ids]
    yC = d_sva[ids_sva]

    n_control_runs = 5
    yA_ctrl = np.zeros((4, d_mem.shape[-1]))
    yB_ctrl, yC_ctrl = np.zeros((4, n_control_runs)), np.zeros((4, n_control_runs))

    for i, k in enumerate([5, 10, 15, 20]):

        agg_func = np.mean
        yA_ctrl[i, :] = aggregate_over(d_mem, labels, f"top{k}-{target_label}-ctrl", match="substring", agg_func=agg_func)
        yB_ctrl[i, :] = d_ppl[select_conditions(labels, f"top{k}-{target_label}-ctrl", match="substring"), ...]
        yC_ctrl[i, :] = d_sva[select_conditions(labels_sva, f"top{k}-{target_label_sva}-ctrl", match="substring"), ...]

    return yA, yA_ctrl, yB, yB_ctrl, yC, yC_ctrl


# %%
def plot_errobars(ax, y, xoffset, color, label):

    y_m = np.median(y, axis=1)
    ci = np.zeros(shape=(2, y.shape[0]))

    # estimate ci per x-axis level
    for k in range(len(y)):
        bs = bootstrap((y[k, :],), axis=0, statistic=np.median, confidence_level=0.95, n_resamples=10000)
        ci[0, k] = bs.confidence_interval.low
        ci[1, k] = bs.confidence_interval.high
    
    ci[0, :] = y_m - ci[0, :]   # subtract lower from median
    ci[1, :] = ci[1, :] - y_m   # subtract higher from median
    
    ax.errorbar(x=np.arange(1, len(y)+1)-xoffset, y=y_m, yerr=ci, color=color, fmt='--', label=label)

    return ax


# %%
def plot_bar(ax, y, xoffset, barwidth, color, hatch=None, label=None):

    ax.bar(np.arange(1, len(y)+1)-xoffset, height=y, color=color, hatch=hatch, width=barwidth, label=label)

    return ax    

# %%
def plot_data_points(ax, y):
    
    xoffset = 0.2

    for i in range(y.shape[-1]):
        yjitter = np.random.RandomState(1234*i).uniform(-0.08, 0.08, len(y))
        ax.scatter(((np.arange(len(y))+1)+xoffset)+yjitter, y[:, i], s=20, color="tab:gray", ec="white", zorder=2)

    return ax

# %%
def despine_ax_add_grid(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(visible=True, which="both", linewidth=0.5)

    return ax


# %%
def despine_axes_set_grid(axes):

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

        a.yaxis.set_minor_locator(AutoMinorLocator(2))
        a.grid(visible=True, which="both", linewidth=0.5)


# %%
def get_ci(y):

    y_m = np.mean(y)

    ci = np.zeros((2, 1))
    bs = bootstrap((y,), axis=0, statistic=np.mean, confidence_level=0.95, n_resamples=10000)
    ci[0] = bs.confidence_interval.low
    ci[1] = bs.confidence_interval.high
        
    ci[0] = y_m - ci[0]   # subtract lower from median
    ci[1] = ci[1] - y_m   # subtract higher from median

    return y_m, ci


# %%
def plot_box_data(ax, xloc, y, horizontal, m='o', c=None, a=0.6, label=None):

    if c is None:
        color = "lightsteelblue"
    else:
        color = c


    xjitter = np.random.uniform(-0.1, 0.1, y.size)
    x = xloc + xjitter
    if horizontal:
        y = xloc + xjitter
        x = y

    ax.scatter(x, y, s=9, marker=m, color=color, alpha=a, zorder=0, label=label)
    ax.boxplot(y, positions=[xloc], widths=0.3, notch=True, vert=~horizontal, showfliers=False, bootstrap=10000,
                medianprops={"linewidth": 1.5, "color": "tab:red"})

    return ax

# %%

def make_memory_plot(axes0, data1, data2, data3, data4, fs):

    y1_mem, y1_mem_ctrl = data1
    y2_mem, y2_mem_ctrl = data2
    y3_mem, y3_mem_ctrl = data3

    unablated = data4

    clrs_deep = list(sns.color_palette("deep").as_hex())

    offset = 0.17
    for i, k in enumerate([1, 2, 3, 4]):
        l, l2 = None, None
        if i == 0:
            l = "Layer-matched\nrandom heads"
            l2 = "Targeted ablation"

        #
        plot_box_data(axes0[0], k-offset, y1_mem[i, :], horizontal=False, c=clrs_deep[2], a=0.2,)
        ax = plot_box_data(axes0[0], k+offset, y1_mem_ctrl[i, :], m='^', c="silver", horizontal=False, a=0.4)

        plot_box_data(axes0[1], k-offset, y2_mem[i, :], horizontal=False, c=clrs_deep[0], a=0.2, label=l2)
        plot_box_data(axes0[1], k+offset, y2_mem_ctrl[i, :], m='^', c="silver", horizontal=False, a=0.4, label=l)

        plot_box_data(axes0[2], k-offset, y3_mem[i, :], horizontal=False, c=clrs_deep[1], a=0.3)
        plot_box_data(axes0[2], k+offset, y3_mem_ctrl[i, :], m='^', c="silver", horizontal=False, a=0.4)


    bs = bootstrap((unablated,), axis=0, statistic=np.median, confidence_level=0.95, n_resamples=10000)
    ci = (bs.confidence_interval.low, bs.confidence_interval.high)


    for a in axes0:
        a.hlines(y=100, xmin=a.get_xlim()[0], xmax=a.get_xlim()[1], linestyle="--", color="black", label="Score indicating\nno memory")
        a.hlines(y=np.median(unablated), xmin=a.get_xlim()[0], xmax=a.get_xlim()[1], linestyle="-.", color="black", label="Unablated model")
        a.fill_between(x=np.arange(a.get_xlim()[0], a.get_xlim()[1]), y1=ci[0], y2=ci[1], alpha=0.2, color='black', zorder=0)

    axes0[0].set_title("Matching heads", fontweight="semibold", color=clrs.green)
    axes0[1].set_title("Post-match heads", fontweight="semibold", color=clrs.blue)
    axes0[2].set_title("Recent-tokens heads", fontweight="semibold", color=clrs.orange)

    for a in axes0:
        a.set_xticks(range(1, 5))
        a.set_xticklabels([f"{k}" for k in [5, 10, 15, 20]], fontsize=fs)
    
    axes0[0].tick_params(axis="y", labelsize=fs)

    axes0[0].set_ylabel("Repeat surprisal (%)", fontsize=fs)
    axes0[1].legend(fontsize=11)

    # set symbol opacity:
    leg = axes0[1].legend()
    leg.legendHandles[0].set_ec("gray")
    leg.legendHandles[0].set_alpha(1)
    leg.legendHandles[0].set_fc("none")
    leg.legendHandles[1].set_alpha(1)
    leg.legendHandles[1].set_color("gray")



    despine_axes_set_grid(axes0)

    axes0[0].annotate('', xy=(-1.1, 20), xytext=(-1.1, 100),                   
                        arrowprops=dict(arrowstyle='<-', color="tab:gray"),  
                        annotation_clip=False)

    axes0[0].text(x=-1.5, y=60, s='Worse memory', fontsize=13, color="tab:gray", rotation=90, va="center")

    return axes0

# %%
def generate_plot(datadir, which):

    csvfiles, jsonfiles = construct_filnames()

    dfs = load_memory_ppl_files(datadir=datadir, csvfiles=csvfiles, jsonfiles=jsonfiles)

    y_mem, y_ppl, xlabels = compute_repeat_surprisals(dataframes=dfs)

    sva = load_sva_files(datadir=datadir)
    y_sva, sva_xlabels = get_sva_arrays_from_df(df=sva)

    # read in individual conditions
    y4_mem, y4_mem_ctrl, y4_ppl, y4_ppl_ctrl, y4_sva, y4_sva_ctrl = prepare_data(y_mem, y_ppl, y_sva, xlabels, "copying-and-matching", sva_xlabels, "copying-and-matching")


    y5_ids = select_conditions(xlabels, f"induction-matching-intersect", match="exact").tolist()[0]
    y5_sva_ids = select_conditions(sva_xlabels, f"induction-matching-intersect", match="exact").tolist()[0]
    y5_mem  = y_mem[y5_ids, :]
    y5_ppl= y_ppl[y5_ids], 
    y5_sva = y_sva[y5_sva_ids]

    y6_ids = select_conditions(xlabels, f"matching-bottom5", match="exact").tolist()[0]
    y6_sva_ids = select_conditions(sva_xlabels, f"matching-bottom5", match="exact").tolist()[0]
    y6_mem  = y_mem[y6_ids, :]
    y6_ppl= y_ppl[y6_ids], 
    y6_sva = y_sva[y6_sva_ids]


    # ===== PLOT ==== # 
    #savedir = args.savedir

    agg_func2 = np.mean

    # ===== FIGURE 1 ===== #
    # top row
    if which == "memory_fig":
        
        y1_mem, y1_mem_ctrl, _, _, _, _ = prepare_data(y_mem, y_ppl, y_sva, xlabels, "matching", sva_xlabels, "matching")
        y2_mem, y2_mem_ctrl, _, _, _, _ = prepare_data(y_mem, y_ppl, y_sva, xlabels, "copying", sva_xlabels, "copying")
        y3_mem, y3_mem_ctrl, _, _, _, _ = prepare_data(y_mem, y_ppl, y_sva, xlabels, "previous", sva_xlabels, "previous")
        
        unabl_ids = select_conditions(xlabels, "unablated", match="exact").tolist()[0]
        unabl_mem = y_mem[unabl_ids]

        fig0, axes0 = plt.subplots(1, 3, figsize=(8, 3.5), sharey="all")

        fs=14
        _ = make_memory_plot(axes0, 
                                (y1_mem, y1_mem_ctrl), 
                                (y2_mem, y2_mem_ctrl), 
                                (y3_mem, y3_mem_ctrl),
                                unabl_mem,
                                fs=fs)

        fig0.supxlabel("Number of heads ablated (top-k)", fontsize=fs)
        #fig0.suptitle("Effect of targeted attention head ablations on short-term memory")
        plt.tight_layout()

    return fig0



# %%
def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--which", type=str)
    parser.add_argument("--savedir", type=str)

    # for testing
    input_args = ["--datadir", "C:\\users\\karmeni1\\project\\lm-mem\\data\\ablation\\zero_attn\\topk",
                   "--which", "memory_fig",
                   "--savedir", "C:\\users\\karmeni1\\project\\lm-mem\\fig\\ablation\\topk"]

    if input_args:
        args = parser.parse_args(input_args)
    elif input_args is None:
        args = parser.parse_args()

    set_manuscript_style()

    if args.which == "memory_fig":

        fig = generate_plot(datadir=args.datadir, which="memory_fig")
        plt.show()
        if args.savedir:
            fn = os.path.join(args.savedir, "targeted_ablation_memory")

            save_png_pdf(fig, fn)

    else:
        figsize=(3, 4)

        # top row
        fig2, ax = plt.subplots(1, 1, figsize=figsize, sharey="all")

        offset = 0.2
        for i, k in enumerate([1, 2, 3, 4]):
            l = None
            if i == 0:
                l = "Layer-matched random heads"
            plot_box_data(ax, k-offset, y4_mem[i, :], horizontal=False, c="tab:blue", a=0.2,)
            ax = plot_box_data(ax, k+offset, y4_mem_ctrl[i, :], c="lightgrey", horizontal=False, a=0.8, label=l)

        ax.hlines(y=100, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle="--", color="darkgrey", label="No memory")

        ax.set_ylabel("Repeat surprisal (%)")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([f"Top-{k}" for k in [5, 10, 15, 20]])
        ax.set_title("Matching & post-matching heads")
        ax.legend()
        despine_ax_add_grid(ax)

        fig2.supxlabel("Ablated heads", fontsize=11)
        fig2.suptitle("")
        plt.tight_layout()
        plt.show()

        fig2.savefig(os.path.join(savedir, "matching_and_post-matching_memory.png"), dpi=300)


        fig3, axes1 = plt.subplots(1, 3, figsize=figsize, sharey="all")

        y1_ppl_ctrl_m = agg_func2(y1_ppl_ctrl, axis=1)  # average over k-seeds
        ax = plot_bar(axes1[0], y=y1_ppl, xoffset=0.2, color="tab:blue", barwidth=0.4, label=None)
        ax = plot_bar(axes1[0], y=y1_ppl_ctrl_m, xoffset=-0.2, barwidth=0.4, color="tab:blue", hatch="//", label="Random heads")
        ax = plot_data_points(axes1[0], y1_ppl_ctrl)

        y2_ppl_ctrl_m = agg_func2(y2_ppl_ctrl, axis=1)  # average over k-seeds
        ax = plot_bar(axes1[1], y=y2_ppl, xoffset=0.2, color="tab:orange", barwidth=0.4, label=None)
        ax = plot_bar(axes1[1], y=y2_ppl_ctrl_m, xoffset=-0.2, color="tab:orange", barwidth=0.4, hatch="//", label="Random heads")
        ax = plot_data_points(axes1[1], y2_ppl_ctrl)
        
        y3_ppl_ctrl_m = agg_func2(y3_ppl_ctrl, axis=1)  # average over k-seeds
        ax = plot_bar(axes1[2], y=y3_ppl, xoffset=0.2, color="tab:green", barwidth=0.4, label=None)
        ax = plot_bar(axes1[2], y=y3_ppl_ctrl_m, xoffset=-0.2, color="tab:green", barwidth=0.4, hatch="//", label="Random heads")
        ax = plot_data_points(axes1[2], y3_ppl_ctrl)

        axes1[0].legend()
        axes1[0].set_ylabel("Test-set loss")
        fig3.supxlabel("Ablated heads", fontsize=10)
        fig3.suptitle("Language modeling")

        despine_axes_set_grid(axes1)

        plt.tight_layout()
        fig3.savefig(os.path.join(savedir, "copying-vs-previous_ppl.png"), dpi=300)


        fig4, axes2 = plt.subplots(1, 3, figsize=figsize, sharey="all")

        y1_sva_ctrl_m = agg_func2(y1_sva_ctrl, axis=1)  # average over k-seeds
        ax = plot_bar(axes2[0], y=y1_sva, xoffset=0.2, color="tab:blue", barwidth=0.4, label=None)
        ax = plot_bar(axes2[0], y=y1_sva_ctrl_m, xoffset=-0.2, barwidth=0.4, color="tab:blue", hatch="//", label="Random heads")
        ax = plot_data_points(axes2[0], y1_sva_ctrl)

        y2_sva_ctrl_m = agg_func2(y2_sva_ctrl, axis=1)  # average over k-seeds
        ax = plot_bar(axes2[1], y=y2_sva, xoffset=0.2, barwidth=0.4, color="tab:orange", label=None)
        ax = plot_bar(axes2[1], y=y2_sva_ctrl_m, xoffset=-0.2, barwidth=0.4, color="tab:orange", hatch="//", label="Random heads")
        ax = plot_data_points(axes2[1], y2_sva_ctrl)
        
        y3_sva_ctrl_m = agg_func2(y3_sva_ctrl, axis=1)  # average over k-seeds
        ax = plot_bar(axes2[2], y=y3_sva, xoffset=0.2, barwidth=0.4, color="tab:green", label=None)
        ax = plot_bar(axes2[2], y=y3_sva_ctrl_m, xoffset=-0.2, barwidth=0.4, color="tab:green", hatch="//", label="Random heads")
        ax = plot_data_points(axes2[2], y3_sva_ctrl)

        axes2[0].set_ylabel("Error rate (%)")
        axes2[0].legend()
        fig4.supxlabel("Ablated heads", fontsize=10)
        fig4.suptitle("Subject-verb agreement")

        despine_axes_set_grid(axes2)

        plt.tight_layout()
        fig4.savefig(os.path.join(savedir, "copying-vs-previous_sva.png"), dpi=300)

        plt.show()


        fig4, ax = plt.subplots(1, 1, figsize=(8, 3))

        plot_box_data(ax, 1, y4_mem[3, :])
        plot_box_data(ax, 2, y5_mem)
        plot_box_data(ax, 3, y3_mem[2, :], c="tab:green", a=0.3)
        plot_box_data(ax, 4, y3_mem[3, :], c="tab:green", a=0.3)
        plot_box_data(ax, 5, y6_mem, c="tab:green", a=0.3)

        ax.set_xlabel("Repeat surprisal (%)")
        ax.set_ylabel("Ablated head(s)")

        xticklabels = ["Top-20 copying-and-matching\n(attn to matching token $t$ and $t+1$)", 
                    "L10.H8\n(is both copying & matching, 1 head)", 
                    "Top 15 matching", 
                    "Top 20 matching",
                    "Top 15-20 matching\n(5 heads)"]
        ax.set_yticklabels(xticklabels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(visible=True, which="both", linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        ax.set_title("Memory performance for additional control ablations")
        plt.tight_layout()
        plt.show()
        fig4.savefig(os.path.join(savedir, "memory_extra.png"), dpi=300)


        
        attn_dict = dict(np.load("C:\\users\\karmeni1\\project\\lm-mem\\data\\ablation\\attention_weights_gpt2_colon-colon-p1.npz"))
        toi = [13]
        topk = [5, 10, 15, 20]
        toi_name = "matching"
        seeds = {54321: 0, 99999: 1, 56789: 2, 98765: 3, 11111: 4}

        set_manuscript_style()

        def plot_head_selection1(fig, ax, layer_head_dict, vals, cmap):

            sel_dict = layer_head_dict
            x = np.zeros((12, 12))
            for i in list(sel_dict.keys()):
                if sel_dict[i]:
                    x[np.array(sel_dict[i]), i] = True

            im = ax.imshow(vals, cmap=cmap, vmin=0)

            ax.set_ylabel("Head")
            ax.set_xlabel("Layer")

            lays = [(i-0.35, h-0.15) for i in sel_dict.keys() if sel_dict[i] for h in sel_dict[i]]
            for xy in lays:
                rect1 = patches.Rectangle(xy, 0.35, 0.35, linewidth=1.5, facecolor="tab:red")
                ax.add_patch(rect1)

            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.set_ylabel("Attention weight")

            return ax

        def plot_head_selection(seed):

            fig, ax = plt.subplots(1, 4, figsize=(12, 4))

            for j, k in enumerate(topk):

                sel_dict, ctrl_dict, vals = find_topk_attn(attn_dict["data"], k, tokens_of_interest=toi, seed=seed)

                x = np.zeros((12, 12))
                x2 = np.zeros((12, 12))
                for i in list(sel_dict.keys()):
                    if sel_dict[i]:
                        x[np.array(sel_dict[i]), i] = True
                        x2[np.array(ctrl_dict[i]), i] = True

                im = ax[j].imshow(vals, cmap="Greens", vmin=0)

                lays = [(i-0.35, h-0.15) for i in sel_dict.keys() if sel_dict[i] for h in sel_dict[i]]
                for xy in lays:
                    rect1 = patches.Rectangle(xy, 0.35, 0.35, linewidth=1.5, facecolor="tab:red")
                    ax[j].add_patch(rect1) 

                lays = [(i+0.08, h-0.15) for i in ctrl_dict.keys() if ctrl_dict[i] for h in ctrl_dict[i]]
                for xy in lays:
                    rect2 = patches.Rectangle(xy, 0.35, 0.35, facecolor="black")
                    ax[j].add_patch(rect2) 

                ax[j].set_title(f"Top-{k}")
                ax[0].set_ylabel("Head")

                cax = ax[-1].inset_axes([1.02, 0, 0.03, 1])
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.set_ylabel("Attention weight")

            ax[0].legend([rect1, rect2], ["Selected", "Control"], title="Head", loc="lower left", bbox_to_anchor=(0, -0.5))

            fig.supxlabel("Layer")
            plt.suptitle(f"{toi_name.capitalize()}-token and control heads (seed {seed})", fontsize=16)

            return fig
        

        for key in seeds.keys():
            fig = plot_head_selection(seed=key)
            plt.show()
            fn = f"C:\\users\\karmeni1\\project\\lm-mem\\fig\\ablation\\topk\\heads_{toi_name}-seed-{key}_B.png"
            print(f"Saving {fn}")
            fig.savefig(fn, dpi=300)


        fig5, ax = plt.subplots(1, 1, figsize=(6, 6))
        lh_dict, _, vals = find_topk_attn(attn_dict["data"], 20, tokens_of_interest=[13, 14], seed=12345)
        ax = plot_head_selection1(fig5, ax, lh_dict, vals, cmap="Greys")
        ax.set_title("Top-20 copying and matching heads\n(attending to matching token at t and t+1)")
        plt.tight_layout()
        plt.show()
        fig5.savefig(os.path.join(savedir, "top20_copying-and-matching_heads.png"), dpi=300)


        fig6, ax = plt.subplots(1, 1, figsize=(6, 6))
        top15_matching, _, _ = find_topk_attn(attn_dict["data"], topk=15, tokens_of_interest=[13], seed=12345)
        top20_matching, _, vals = find_topk_attn(attn_dict["data"], topk=20, tokens_of_interest=[13], seed=12345)
        top15labels, top20labels = from_dict_to_labels(top15_matching), from_dict_to_labels(top20_matching)
        resid_labels = list(set(top20labels) - set(top15labels))

        lh_dict = from_labels_to_dict(resid_labels)

        ax = plot_head_selection1(fig6, ax, lh_dict, vals, cmap="Greens")
        ax.set_title("Top 15-20 matching heads")
        plt.tight_layout()
        plt.show()
        fig6.savefig(os.path.join(savedir, "top15-20_matching_heads.png"), dpi=300)



        abl_dicts = {"copying": [14, 16, 18], "recent": [42, 43, 44], "matching": [13]}

        k = 20
        d, v = [], []
        for key in abl_dicts.keys():
            sel_dict, _, vals = find_topk_attn(attn_dict["data"], k, tokens_of_interest=abl_dicts[key], seed=12345)
            
            d.append(sel_dict)
            v.append(vals)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        offsets = [-0.25, 0, +0.25]
        labels = ["Copying", "Recent-tokens", "Matching"]

        for i, x in enumerate(d):
            counts = [len(x[key]) for key in x]    

            ax.bar(np.arange(len(counts))+offsets[i], height=counts, width=0.25, label=labels[i])
        
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(np.arange(len(counts))+1)

        ax.set_title(f"Distribution of head types (top-{k}) across GPT-2 layers", fontsize=16)
        ax.set_xlabel("Layer", fontsize=14)
        ax.set_ylabel("Nr. heads", fontsize=14)
        ax.legend(title="Head type")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(visible=True, linewidth=0.5)

        plt.show()
    #fig.savefig(os.path.join(savedir, "head_distribution.png"), dpi=300)


    return 0

# %%
if __name__ == "__main__":

    main()
# %%
