import os, sys
sys.path.append(os.environ["PROJ_ROOT"])

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import bootstrap, median_abs_deviation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
from src.wm_suite.wm_ablation import find_topk_attn
from src.wm_suite.viz.func import filter_and_aggregate
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")

ABLATION_FILENAME_CODES = [str(i) for i in range(12)] + ["01", "23", "56", "711", "0123", "56711", "all"]


def check_datadir(datadir: str) -> str:

    if "WMS_DATADIR" in os.environ.keys():
        datadir = os.environ["WMS_DATADIR"]
        logging.info(f"WMS_DATADIR environment variable found using {datadir}")
    elif datadir is None:
        raise ValueError("Data directory not set up, use --datadir argument or set the"
                         "environment variable 'WMS_DATADIR' to appropriate path")
    else:
        logging.info(f"Using {datadir} as data directory")

    return datadir


def get_data(fn: str) -> Tuple[Dict, np.ndarray]:
    
    print(f"Loading {fn}")
    b = dict(np.load(fn), allow_pickle=True) 
    a = np.stack(b['data']) # shape = (seq, token, head, layer)
    
    return a, b


def get_mean_se(x: np.ndarray, axis: Tuple) -> Tuple[float, float]:
    """
    Parameters:
    ----------
    x : np.ndarray, shape = (n_samples, n_timesteps, n_heads, n_layers)
        the data array containing wesihgs
    axis: tuple
        axis over which to average

    Returns:
    -------
    m : np.ndarray, shape = (n_layers,)
        mean
    se : np.ndarray, shape = (n_layers,)
        the standard error of the mean
    """
    logging.info("Call to get_mean_se(), assuming dimensions:")
    logging.info(f"n_samples (0): {x.shape[0]}\n n_heads (1): {x.shape[1]}\n n_layers (2): {x.shape[2]}")

    logging.info(f"Taking mean over dimensions {axis} of lengths {x.shape[axis[0]]} and {x.shape[1]}")
    m = np.mean(x, axis=axis)

    # first average over sequences, then get SEM over heads for each layer (axis=1)
    mean_per_head_layer = np.mean(x, axis=0)
    sem_per_layer = median_abs_deviation(mean_per_head_layer, axis=0)
    
    return m, sem_per_layer


def plot_attention(ax: mpl.axes.Axes, x: np.ndarray, token_ids: np.ndarray, labels: List):
    """
    Parameters:
    ----------
    ax: mpl.Axes
        axis (single dimensional) onto which to plot attention weigths
    x: np.ndarray, shape = (layers, timesteps, sequences, heads)

    """

    # loop over selected target tokens that we're attending to
    for t, l in zip(token_ids, labels):

        # average acrross heads and across sequences
        attn_at_timestep = x[:, t, :, :]  # shape = (sequences, heads, layers)

        # now compute mean over heads and layers (0, 1)
        m, se = get_mean_se(attn_at_timestep, axis=(0, 1))

        ax.plot(m, '--o', label=l)
        ax.fill_between(np.arange(m.shape[-1]), y1=m+se, y2=m-se, alpha=0.2)

    fontsize=16
    ax.set_ylabel('Avg. attention weight\n(across heads & sequences)', fontsize=fontsize)
    ax.set_xlabel('Layer', fontsize=fontsize)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(np.arange(12)+1, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def attn_amplitude_plot(axes, x: np.ndarray, d: Dict, target_id=13, query_id=45, seed=12345):

    # set ylabels for legend
    ylabels = [s.strip('Ġ') for s in d['tokens'][0]]
    
    ids1 = (target_id+5, target_id+3, target_id+1, target_id)
    labels = [f"third noun [{ids1[0]+1}]", 
              f"second noun [{ids1[1]+1}]", 
              f"first noun [{ids1[2]+1}]", 
              f"cue token (':') [{ids1[3]+1}]"]

    #ax1, ax2, ax3 = plt.subplot(gs[0, :]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
    for a in axes:
        a.set_prop_cycle('color', plt.cm.YlGnBu(np.linspace(0.4, 0.9, len(labels))))
        
    ax1, ax2 = axes


    plot_attention(ax1, x, ids1, labels)
    #ax1.set_title("Attention to tokens in first list", fontsize=12)    
    
    query_ids = (query_id-1, query_id-2, query_id-3)
    labs = [f"{ylabels[q]} (t-{i+1}) [{q}]" for i, q in enumerate(query_ids)]
    plot_attention(ax2, x, query_ids, labs)
    #ax2.set_title("Attention to immediately preceding tokens", fontsize=12)

    return ax1, ax2


def attn_amplitude_plot_control_tokens(axes, x: np.ndarray, d: Dict, colors, seed=12345):

    ylabels = [s.strip('Ġ') for s in d['tokens'][0]]

    np.random.seed(seed)
    start_timestep = np.random.choice(np.arange(24, 35), 1)[0]      # select a random token
    ints = [start_timestep-1, start_timestep, start_timestep+1]  # now select one before and one after
    labs = [f"{ylabels[i]} [{i+1}]" for i in ints]

    sel = np.zeros(shape=x.shape[1], dtype=bool)
    sel[np.array(ints)] = True

    xtmp = np.sum(x[:, sel, ...], 1)
    _, img1, data1 = plot_imshow_and_average(axes, xtmp, c=colors[0])

    axes[0].set_title(f"To random intermediate tokens ({' '.join(labs)})")

    return axes, img1, data1


def plot_schematic(ax, d, colors, squeezed=False):
    
#    first_list = " ".join([s.strip("Ġ") for s in d['tokens'][0][13:20]])
#    ctx = " ".join([s.strip("Ġ") for s in d['tokens'][0][20:35]])
#    second_list = " ".join([s.strip("Ġ") for s in d['tokens'][0][-11:-1]])

    cue1 = d['tokens'][0][13].strip("Ġ")
    cue2 = d['tokens'][0][45].strip("Ġ")

    labels = ["Mary wrote down a list of words", cue1, "$N_1, N_2, N_3$", 
                "[...] when she got back she read", "the list again",  cue2]

    xmax=39
    if squeezed:
        xpos = [0*xmax, 0.346*xmax, 0.37*xmax, 0.50*xmax, 0.85*xmax, xmax]
        text_fs, annot_fs = 16, 16
    else:
        xpos = [0*xmax, 0.33*xmax, 0.37*xmax, 0.51*xmax, 0.85*xmax, xmax]    
        text_fs, annot_fs = 15, 15

    ypos = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
    xarrow_from = [39, 39, 39]
    xarrow_to = [16.5, 35.5, 13.5]

    if squeezed:
        fractions = [0.08, 0.26, 0.1]
        lw = 2
    else:
        fractions = [0.1, 0.26, 0.15]
        lw = 1.7

    ax.set_axis_off()

    for i, tk in enumerate(labels):

        c, fc, fcalpha = "tab:gray", "None", None
        if i in [2, 4]:
            c, fc, fcalpha = "black", "tab:gray", 0.3
        elif i in [1, 5]: 
            c, fc, fcalpha = "black", "tab:red", 0.3

        if cue1 != cue2:
            if i in [1]:
                c, fc, fcalpha = "black", "tab:red", 0.3
            if i in [5]:
                c, fc, fcalpha = "black", "tab:green", 0.3

        ax.text(xpos[i], ypos[i], tk, color=c, fontsize=text_fs, ha='left',
        bbox=dict(facecolor=fc, alpha=fcalpha, edgecolor='gray', boxstyle='round'))

    for i in range(len(xarrow_from)):

            ax.annotate('', 
                        xy=(xarrow_to[i], 0.75),
                        xytext=(xarrow_from[i], 0.75), 
                        va='center', 
                        ha='center',
                        fontsize=14,
                        arrowprops={'arrowstyle': '->', 
                                    'color': f"{colors[i]}",
                                    "lw": lw,
                                    'ls':'dashed', 
                                    'connectionstyle': f'bar,fraction={fractions[i]}'})

    if not squeezed:
        ax.text(39.3, 0.8, "Query", color="tab:gray", fontsize=annot_fs, ha="left", fontweight="bold")
        ax.text(39.3, 1.8, "Attention", color="tab:gray", fontsize=annot_fs, ha="left", fontweight="bold")

        ax.text(26, 2, "Post-matching heads", color=colors[0], fontsize=annot_fs, ha="center", fontweight="bold")
        ax.text(34.5, 1.3, "Recent-tokens heads", color=colors[1], fontsize=annot_fs, ha="center", fontweight="bold")
        ax.text(18, 2.9, "Matching heads", color=colors[2], fontsize=annot_fs, ha="center", fontweight="bold")

    ax.text(-2.5, 0.35, "LM Input", color="tab:gray", fontsize=annot_fs, ha="center", fontweight="bold")
    ax.set_xticks(np.arange(0, 40))

    return ax


def plot_imshow_and_average(axes, dat:np.ndarray, selection: Dict=None, c:str="#000000", img_txt_c="white"):

    m = np.mean(dat, axis=(0, 1))           # mean across sequences and heads
    se = sem(np.mean(dat, axis=0), axis=0)  # SE over head means

    axes[0].plot(m, '--o', markersize=8, mec='white', color=c)
    axes[0].fill_between(np.arange(dat.shape[-1]), y1=m+se, y2=m-se, color=c, alpha=0.3)

    if c == "#1f77b4":
        cmap = plt.cm.Blues
    elif c == "#ff7f0e":
        cmap = plt.cm.Oranges
    elif c == "#2ca02c":
        cmap = plt.cm.Greens
    elif c == "#000000":
        cmap = plt.cm.Greys
    im1 = axes[1].imshow(np.mean(dat, axis=0), cmap=cmap, aspect='auto', vmin=0, vmax=1)

    if selection:
        lays = [(i-0.45, h-0.5) for i in selection.keys() if selection[i] for h in selection[i]]
        for xy in lays:
            rect = patches.Rectangle(xy, 0.99, 0.99, linewidth=1.3, edgecolor="black", facecolor="none")
            axes[1].add_patch(rect) 

        textc = "white"
        if c == "#2ca02c":
            textc = "black"
        xycoord = [(i, h+0.1) for i in selection.keys() if selection[i] for h in selection[i]]
        xyinds = [(i, h) for i in selection.keys() if selection[i] for h in selection[i]]
        for xy, xyind in zip(xycoord, xyinds):

            value = round(np.mean(dat, axis=0)[xyind[1], xyind[0]], 2)
            texts = f"{value:.2f}".lstrip("0")
            if value == 1.0:
                texts = f"{value:.0f}"
            
            axes[1].text(x=xy[0], y=xy[1], s=texts, ha="center", va="center", c=img_txt_c, fontweight="semibold")


    return axes, im1, (m, se)


def attn_weights_per_head_layer(ax, x: np.ndarray, colors: List, img_text_colors: List, query_id: int, target_id: int, n_tokens_per_window: int):
    """"
    plots 3-by-3 figure with line plots and images

    Parameters:
    ----------
    x : np.ndarray, shape = (samples, timesteps, n_heads, n_layers)

    """

    # token indices from nearest to furthest
    first_list = [target_id + i for i in np.arange(1, n_tokens_per_window*2, 2)]
    preceding = [query_id - i for i in range(1, n_tokens_per_window+1)]

    logging.info(f"Summing attn weights over early tokens: {np.array(first_list)}")
    logging.info(f"Summing attn weights over late tokens: {np.array(preceding)}")

    # sum over selected positions and plot image
    sel = target_id
    xtmp = x[:, sel, ...]

    selection, _, _ = find_topk_attn(x, topk=10, tokens_of_interest=[target_id], seed=12345)
    _, img1, d1 = plot_imshow_and_average(ax[:, 0], xtmp, selection=selection, c=colors[2], img_txt_c=img_text_colors[0])

    # sum over selected positions and plot image
    sel = np.zeros(shape=x.shape[1], dtype=bool)
    sel[np.array(first_list)] = True
    xtmp = np.sum(x[:, sel, ...], 1)

    selection, _, _ = find_topk_attn(x, topk=10, tokens_of_interest=first_list, seed=12345)  # to mark top-10 cells
    _, img2, d2 = plot_imshow_and_average(ax[:, 1], xtmp, selection=selection, c=colors[0], img_txt_c=img_text_colors[1])

    # sum over selected positions and plot image
    sel = np.zeros(shape=x.shape[1], dtype=bool)
    sel[np.array(preceding)] = True
    xtmp = np.sum(x[:, sel, ...], 1)

    selection, _, _ = find_topk_attn(x, topk=10, tokens_of_interest=preceding, seed=12345)  # to mark top-10 cells
    _, img3, d3 = plot_imshow_and_average(ax[:, 2], xtmp, selection=selection, c=colors[1], img_txt_c=img_text_colors[2])

    for a in ax[0, :]:
        a.grid(visible=True, linewidth=0.5)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    return ax, (img1, img2, img3), (d1, d2, d3)


def generate_plot2(datadir, query):

    if query == "colon-colon-p1":
        fn = "attention_weights_gpt2_colon-colon-p1.npz"
        query_idx = 45
        suptitle = ""
        n_tokens_per_window = 3

    elif query == "colon-semicolon-p1":
        fn = "attention_weights_gpt2_colon-semicolon-p1.npz"
        query_idx = 45
        suptitle = ""
        n_tokens_per_window = 3
    
    elif query == "comma-comma-p1":
        fn = "attention_weights_gpt2_comma-comma-p1.npz"
        query_idx = 45  # this is first noun in the list
        suptitle = ""
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ablate-11":
        fn = "attention_weights_gpt2-ablate-10-all_colon-colon-p1-ctxlen1.npz"  # this is 0 indexing
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ablated layer 11)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ablate-2":
        fn = "attention_weights_gpt2-ablate-2-all_colon-colon-p1-ctxlen1.npz"  # this is 0 indexing
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ablated layer 3)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ctxlen3":
        fn = "attention_weights_gpt2_colon-colon-p1-ctxlen3.npz"  # this is 0 indexing
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ctxlen3)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ctxlen4":
        fn = "attention_weights_gpt2_colon-colon-p1-ctxlen4.npz"  # this is 0 indexing
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ctxlen4)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-n1":
        fn = "attention_weights_gpt2_colon-colon-p1.npz"
        query_idx = 45
        suptitle = "Copy and previous-token attention heads in GPT-2 (single-token window)"
        n_tokens_per_window = 1

    elif query == "colon-colon-n2":
        fn = "gpt2_attn_query-n2.npz"
        query_idx = 48
        title_string = ":"

    elif query == "colon-semicolon-n2":
        fn = "attention_weights_gpt2_colon-semicolon-n2.npz"
        query_idx = 48 # this is the position of the second noun in the list
        title_string = "second noun"

    x1, d1 = get_data(os.path.join(datadir, fn))

    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:3]

    # ===== PLOT ===== #

    if query in ["colon-colon-p1", 'colon-semicolon-p1', 'comma-comma-p1']:

        fig = plt.figure(figsize=(12, 6.5))

        gs = GridSpec(3, 3, height_ratios=[0.6, 1, 2.5], figure=fig)

        ax1 = fig.add_subplot(gs[0, :])

        # first row axes
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
        ax4 = fig.add_subplot(gs[1, 2], sharey=ax2)

        # second row axes
        ax5 = fig.add_subplot(gs[2, 0], sharex=ax2)
        ax6 = fig.add_subplot(gs[2, 1], sharex=ax3, sharey=ax5)
        ax7 = fig.add_subplot(gs[2, 2], sharex=ax4, sharey=ax5)

        # for the main figure, font can remain smaller (as the figure itself is lareger), 
        if query in ["colon-colon-p1"]:
            plot_schematic(ax1, d1, colors=clrs, squeezed=False)
        else:
            plot_schematic(ax1, d1, colors=clrs, squeezed=True)

        axes = np.array([[ax2, ax3, ax4], [ax5, ax6, ax7]])

        for a in (ax3, ax4, ax6, ax7):
            plt.setp(a.get_yticklabels(), visible=False)
        for a in (ax2, ax3, ax4):
            plt.setp(a.get_xticklabels(), visible=False)

        #axes[0, 0].set_title("1")
        #axes[0, 1].set_title("2")
        #axes[0, 2].set_title("3")
        #axes[1, 0].set_title("3")
        #axes[1, 1].set_title("4")
        #axes[1, 2].set_title("5")

        #plt.show()

    else:

        fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), sharex="all", sharey="row", 
                               gridspec_kw={'height_ratios': [1, 2]},
                               )

    # ==== PLOT FIGURE ===== #
    if query in ["colon-colon-p1"]:
        img_txt_clrs = ["black", "white", "white"]
    else:
        img_txt_clrs = ["black", "black", "white"]
    ax, imgs, data1 = attn_weights_per_head_layer(ax=axes, x=x1, colors=clrs, img_text_colors=img_txt_clrs,
                                                  target_id=13, 
                                                  query_id=query_idx,
                                                  n_tokens_per_window=n_tokens_per_window)

    lfs = 18
    titlefs = 19
    if query in ["comma-comma-p1", "colon-semicolon-p1", "colon-colon-p1-n1"]:
        lfs = 22
        titlefs = 23

    ax[1, 0].set_ylabel("Head", fontsize=lfs)
    ax[0, 0].set_ylabel("Avg. attention\nweight", fontsize=lfs)

    # suptitles
    #ax[0, 0].set_title("Attention to tokens in first list", fontsize=16)
    #ax[0, 1].set_title("Attention to previous tokens", fontsize=16)

    for a in ax[0, :]:
        a.set_ylim([0, 0.5])
        a.set_yticks([0, 0.5])
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.grid(visible=True, which="both", linewidth=0.5)

    # ticks and ticklabels
    for a in ax[1, :]:
        a.set_xticks(np.arange(0, 11, 2))
        a.set_xticklabels(np.arange(1, 12, 2), fontsize=lfs)
        a.yaxis.set_minor_locator(AutoMinorLocator(2))
        a.xaxis.set_minor_locator(AutoMinorLocator(2))

 
    ax[0, 0].tick_params(axis="y", labelsize=lfs)
    ax[1, 0].set_yticks(np.arange(0, 12, 2))
    ax[1, 0].set_yticklabels(np.arange(1, 12, 2), fontsize=lfs)

    # colorbar
    cax = ax[1, 0].inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(imgs[0], ax=ax[1, 0], cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["", "", ""])

    cax = ax[1, 1].inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(imgs[1], ax=ax[1, 1], cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["", "", ""])

    cax = ax[1, 2].inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(imgs[2], ax=ax[1, 2], cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels([0, 0.5, 1])
    cbar.ax.set_ylabel("Attention weight", fontsize=lfs-3)
    cbar.ax.tick_params(labelsize=lfs-3)

    fig.supxlabel("Layer", fontsize=lfs)
    fig.suptitle(suptitle, fontsize=titlefs)
    fig.tight_layout()

    # ===== CONTROL FIGURE ===== #

    fig_, axes2 = plt.subplots(2, 1, figsize=(6, 4.5), sharex="col",
                               gridspec_kw={"height_ratios": [1, 2]})

    axes2, img2_, data2 = attn_amplitude_plot_control_tokens(axes2, x1, d1, colors=["#000000"], seed=12345)
    #axes2.legend(title="Target token", fontsize=12, title_fontsize=12)

    lfs=16

    axes2[0].set_ylim([0, 0.5])
    axes2[0].set_yticks([0, 0.25, 0.5])
    axes2[0].set_yticklabels([0, '', 0.5])

    axes2[1].set_yticks(np.arange(0, 12, 2))
    axes2[1].set_yticklabels(np.arange(1, 12, 2), fontsize=lfs)

    fig_.supxlabel("Layer", fontsize=lfs)
    axes2[0].set_ylabel("Avg. attention\nweight", fontsize=lfs)
    axes2[1].set_ylabel("Head", fontsize=lfs)

    for a in axes2:
        a.tick_params(axis="y", labelsize=lfs)

    for a in axes2:
        a.set_xticks(np.arange(0, 11, 2))
        a.set_xticklabels(np.arange(1, 12, 2), fontsize=lfs)

    # despine and gridlines
    axes2[0].grid(visible=True, linewidth=0.5)
    axes2[0].spines['top'].set_visible(False)
    axes2[0].spines['right'].set_visible(False)

    cax = axes2[1].inset_axes([1.04, 0.1, 0.04, 0.8])
    cbar = fig_.colorbar(img2_, ax=axes2[1], cax=cax)
    cbar.ax.set_ylabel("Attention weight", rotation=90, fontsize=lfs)
    cbar.ax.tick_params(labelsize=lfs)

    
    fig_.suptitle("GPT-2 attention over random tokens in intervening context", fontsize=15)
    fig_.tight_layout()

    # format data in a csv
    datarec_ax0 = {f"layer-{i+1}": v for i, v in enumerate(data1[0][0])}   # to nouns
    datarec_ax1 = {f"layer-{i+1}": v for i, v in enumerate(data1[1][0])}   # to nouns
    datarec_fig0 = {f"layer-{i+1}": v for i, v in enumerate(data2[0])}   # to nouns
    data = pd.DataFrame.from_dict([datarec_ax0, datarec_ax1, datarec_fig0]).T
    data.columns = ['distant', 'recent', 'intermediate']


    return fig, fig_, data


def save_png_pdf(fig, savename: str):

    savefn = os.path.join(savename + ".png")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, dpi=300, format="png")

    savefn = os.path.join(savename + ".pdf")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, format="pdf", transparent=True, bbox_inches="tight")

    return 0


def main(input_args=None):

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, choices=["all", "main_fig", "control_fig", "single_token_fig"])
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    datadir = check_datadir(args.datadir)


    if args.which is None:
        raise ValueError("Please specify which figures to make by providing '--which' argument")

    if args.which == "all":
        
        main_fig = True
        control_fig = True
        single_token_fig = True
    
    else:

        main_fig = True if args.which == "main_fig" else False
        control_fig = True if args.which == "control_fig" else False
        single_token_fig = True if args.which == "single_token_fig" else False


    with plt.style.context("seaborn-ticks"):

        if main_fig:

            fig1, fig1_, data = generate_plot2(datadir=datadir, query="colon-colon-p1")

            if args.savedir:
                save_png_pdf(fig1, os.path.join(args.savedir, "gpt2_attn_colon-colon-p1"))
                save_png_pdf(fig1_, os.path.join(args.savedir, "gpt2_attn_colon-colon-p1_control"))

                fn = os.path.join(args.savedir, "gpt2_attn_colon-colon-p1" + ".csv")
                logging.info(f"Saving {fn}")
                data.to_csv(fn, sep='\t')

        # ===== CONTROL FIGURES: SWAP QUERY TOKENS ==== #
        if control_fig:

            fig2, _, data = generate_plot2(datadir=datadir, query="colon-semicolon-p1")

            if args.savedir:

                savename = "gpt2_attn_colon-semicolon-p1"
                save_png_pdf(fig2, os.path.join(args.savedir, savename))

                fn = os.path.join(args.savedir, savename + ".csv")
                logging.info(f"Saving {fn}")
                data.to_csv(fn, sep='\t')


            fig3, _, data = generate_plot2(datadir=datadir, query="comma-comma-p1")
            if args.savedir:

                savename = "gpt2_attn_comma-comma-p1"
                save_png_pdf(fig3, os.path.join(args.savedir, savename))
                
                fn = os.path.join(args.savedir, savename + ".csv")
                logging.info(f"Saving {fn}")
                data.to_csv(fn, sep='\t')

        # single token plot
        if single_token_fig:
            fig4, fig4_, data = generate_plot2(datadir=datadir, query="colon-colon-p1-n1")
            if args.savedir:
                save_png_pdf(fig4, os.path.join(args.savedir, "gpt2_attn_colon-colon-p1-n1"))

        # ===== CONTROL FIGURES: ABLATED MODEL ==== #
        #fig5, fig5_, data = generate_plot2(datadir=datadir, query="colon-colon-p1-ablate-11")
        #if args.savedir:
        #    save_png_pdf(fig5, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1"))
        #    save_png_pdf(fig5_, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1_control"))

        #    fn = os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1")
        #    logging.info(f"Saving {fn}")
        #    data.to_csv(fn, sep='\t')

        #fig6, fig6_, data = generate_plot2(datadir=datadir, query="colon-colon-p1-ablate-2")
        #if args.savedir:
        #    save_png_pdf(fig6, os.path.join(args.savedir, "gpt2-ablate-2_attn_colon-colon-p1"))
        #    save_png_pdf(fig6_, os.path.join(args.savedir, "gpt2-ablate-2_attn_colon-colon-p1_control"))

        #    fn = os.path.join(args.savedir, "gpt2-ablate-2_attn_colon-colon-p1")
        #    logging.info(f"Saving {fn}")
        #    data.to_csv(fn, sep='\t')


        # ===== CONTROL FIGURES: LONG CONTEXT ==== #
        #fig7, fig7_ = generate_plot2(datadir=datadir, query="colon-colon-p1-ctxlen3")
        #if args.savedir:
        #    save_png_pdf(fig7, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1-ctxlen3"))
        #    save_png_pdf(fig7_, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1-ctxlen3_control"))
        
        #fig8, fig8_ = generate_plot2(datadir=datadir, query="colon-colon-p1-ctxlen4")
        #if args.savedir:
        #    save_png_pdf(fig8, os.path.join(args.savedir, "gpt2-ablate-0_attn_colon-colon-p1-ctxlen4"))
        #    save_png_pdf(fig8_, os.path.join(args.savedir, "gpt2-ablate-0_attn_colon-colon-p1-ctxlen4_control"))

        plt.show()

    #plt.close("all")

    #with plt.style.context("seaborn-whitegrid"):

    #    fig1, fig2 = generate_plot(datadir=datadir, query="colon-semicolon-p1")

if __name__ == "__main__":

    main()