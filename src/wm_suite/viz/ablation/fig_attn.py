# %%
import os
import numpy as np
import pandas as pd
from scipy.stats import sem
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

from wm_suite.wm_ablation import find_topk_attn
from wm_suite.viz.func import set_manuscript_style
from wm_suite.viz.utils import get_font_config

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("wm_suite.utils")


# %%
def get_data(fn: str) -> Tuple[Dict, np.ndarray]:
    
    logger.info(f"Loading {fn}")
    b = dict(np.load(fn), allow_pickle=True) 
    a = np.stack(b['data']) # shape = (seq, token, head, layer)
    
    return a, b


# %%
def plot_schematic(ax, d, colors, annot_fs, text_fs, squeezed=False):
    
#    first_list = " ".join([s.strip("Ġ") for s in d['tokens'][0][13:20]])
#    ctx = " ".join([s.strip("Ġ") for s in d['tokens'][0][20:35]])
#    second_list = " ".join([s.strip("Ġ") for s in d['tokens'][0][-11:-1]])

    ax.set_ylim(0, 2)

    cue1 = d['tokens'][0][13].strip("Ġ")
    cue2 = d['tokens'][0][45].strip("Ġ")
    
    xmax = 39
    ypos = 0.85
    labels2 = {
        0: {"text": "Mary wrote down a list of words", "xpos": 0*xmax, "ypos": ypos, "c": "tab:gray", "fc": "None", "fcalpha": None},
        1: {"text": cue1, "xpos": 0.32*xmax, "ypos": ypos, "c": "black", "fc": "tab:green", "fcalpha": 0.3},
        2: {"text": "$N_1$", "xpos": 0.35*xmax, "ypos": ypos, "c": "black", "fc": "tab:blue", "fcalpha": 0.3},
        3: {"text": "$,$", "xpos": 0.4*xmax, "ypos": ypos, "c": "black", "fc": "None", "fcalpha": None},
        4: {"text": "$N_2$", "xpos": 0.43*xmax, "ypos": ypos, "c": "black", "fc": "tab:blue", "fcalpha": 0.3},
        5: {"text": "$,$", "xpos": 0.48*xmax, "ypos": ypos, "c": "black", "fc": "None", "fcalpha": None},
        6: {"text": "$N_3$", "xpos": 0.51*xmax, "ypos": ypos, "c": "black", "fc": "tab:blue", "fcalpha": 0.3},
        7: {"text": "...when she got back she read", "xpos": 0.56*xmax, "ypos": ypos, "c": "tab:grey", "fc": "None", "fcalpha": None},
        8: {"text": "the list again", "xpos": 0.86*xmax, "ypos": ypos, "c": "black", "fc": "tab:orange", "fcalpha": 0.3},
        9: {"text": cue2, "xpos": xmax, "ypos": ypos, "c": "black", "fc": "tab:red", "fcalpha": 0.3},
    }

    xmax=39
    if squeezed:
        xpos = [0*xmax, 0.346*xmax, 0.37*xmax, 0.50*xmax, 0.85*xmax, xmax]
        #text_fs, annot_fs = 16, 16
    else:
        xpos = [0*xmax, 0.33*xmax, 0.37*xmax, 0.51*xmax, 0.85*xmax, xmax]    
        #text_fs, annot_fs = 12, 12

    ypos = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35])+0.7
    xarrow_from = [39, 39, 39]
    xarrow_to = [0.44*xmax, 0.91*xmax, 0.32*xmax]

    if squeezed:
        fractions = [0.08, 0.26, 0.1]
        lw = 2
    else:
        fractions = [0.07, 0.36, 0.12]  # (postmatch, recent, matching)
        lw = 1.7

    ax.set_axis_off()

    for i, tkey in enumerate(labels2):

        tkdict = labels2[tkey]
        ax.text(tkdict["xpos"], tkdict["ypos"], tkdict["text"], color=tkdict["c"], fontsize=text_fs, fontweight="semibold", ha='left',
                bbox=dict(facecolor=tkdict["fc"], alpha=tkdict["fcalpha"], edgecolor='gray', boxstyle='round'))

    for i in range(len(xarrow_from)):
        
        if i == 0:
            ypos_to = 2
            ypos_to2 = 2
            arrprops = {'arrowstyle': "-[",
                        'color': f"{colors[i]}",
                        "lw": lw,
                        'ls':'dashed', 
                        'connectionstyle': f'bar,fraction={fractions[i]}'}
        else:
            ypos_to = 1.4
            ypos_to2 = 1.4
            arrprops = {'arrowstyle': "->", 
                        'color': f"{colors[i]}",
                        "lw": lw,
                        'ls':'dashed', 
                        'connectionstyle': f'bar,fraction={fractions[i]}'}

        ax.annotate('', 
                    xy=(xarrow_to[i], ypos_to),
                    xytext=(xarrow_from[i], ypos_to2), 
                    va='center', 
                    ha='center',
                    fontsize=14,
                    arrowprops=arrprops)

    if not squeezed:
        ax.text(39.5, 0.8, "Query", color="tab:gray", fontsize=annot_fs, ha="left", fontweight="bold")
        ax.text(39.5, 1.8, "Attention", color="tab:gray", fontsize=annot_fs, ha="left", fontweight="bold")

        ax.text(26, 3.4, "Postmatch heads", color=colors[0], fontsize=annot_fs, ha="center", fontweight="bold")
        ax.text(34.5, 2.6, "Recent-tokens heads", color=colors[1], fontsize=annot_fs, ha="center", fontweight="bold")
        ax.text(18, 4.2, "Matching heads", color=colors[2], fontsize=annot_fs, ha="center", fontweight="bold")

    ax.text(-2.5, 0.85, "LM Input", color="tab:gray", fontsize=annot_fs, ha="center", fontweight="bold")
    ax.set_xticks(np.arange(0, 40))

    return ax


# %%
def plot_imshow_and_average(axes, dat:np.ndarray, selection: Dict=None, c:str="#000000", img_txt_c="white", img_txt_fs=12):

    m = np.mean(dat, axis=(0, 1))           # mean across sequences and heads
    se = sem(np.mean(dat, axis=0), axis=0)  # SE over head means

    if c == "#1f77b4":
        cmap = plt.cm.Blues
    elif c == "#ff7f0e":
        cmap = plt.cm.Oranges
    elif c == "#2ca02c":
        cmap = plt.cm.Greens
    elif c == "#000000":
        cmap = plt.cm.Greys


    im1 = axes[0].imshow(np.mean(dat, axis=0).T, origin="lower", cmap=cmap, aspect='equal', vmin=0, vmax=1)

    if selection:
        lays = [(h-0.45, i-0.5) for i in selection.keys() if selection[i] for h in selection[i]]
        for xy in lays:
            rect = patches.Rectangle(xy, 0.99, 0.99, linewidth=1.3, edgecolor="black", facecolor="none")
            axes[0].add_patch(rect) 

        textc = "white"
        if c == "#2ca02c":
            textc = "black"

        # annotate cells with attention values    
        xycoord = [(h+0.1, i) for i in selection.keys() if selection[i] for h in selection[i]]
        xyinds = [(h, i) for i in selection.keys() if selection[i] for h in selection[i]]
        for xy, xyind in zip(xycoord, xyinds):

            value = round(np.mean(dat, axis=0)[xyind[0], xyind[1]], 2)
            texts = f"{value:.2f}".lstrip("0")
            if value == 1.0:
                texts = f"{value:.0f}"
            
            axes[0].text(x=xy[0], y=xy[1], s=texts, ha="center", va="center", c=img_txt_c, fontsize=img_txt_fs, fontweight="bold")


    return axes, im1, (m, se)


# %%
def attn_weights_per_head_layer(ax, x: np.ndarray, colors: List, img_text_colors: List, img_text_fs: int, query_id: int, target_id: int, n_tokens_per_window: int):
    """"
    plots 3-by-3 figure with image plots

    Parameters:
    ----------
    x : np.ndarray, shape = (samples, timesteps, n_heads, n_layers)

    """

    # token indices from nearest to furthest
    first_list = [target_id + i for i in np.arange(1, n_tokens_per_window*2, 2)]
    
    # if we
    if target_id == 14:
        first_list = [16, 18, 19]
    preceding = [query_id - i for i in range(1, n_tokens_per_window+1)]

    logger.info(f"Summing attn weights over early tokens: {np.array(first_list)}")
    logger.info(f"Summing attn weights over late tokens: {np.array(preceding)}")

    # sum over selected positions and plot image
    sel = target_id
    xtmp = x[:, sel, ...]

    selection, _, _ = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=[target_id], seed=12345)
    _, img1, d1 = plot_imshow_and_average(ax[:, 0], xtmp, selection=selection, c=colors[2], img_txt_c=img_text_colors[0], img_txt_fs=img_text_fs)

    # sum over selected positions and plot image
    sel = np.zeros(shape=x.shape[1], dtype=bool)
    sel[np.array(first_list)] = True
    xtmp = np.sum(x[:, sel, ...], 1)

    selection, _, _ = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=first_list, seed=12345)  # to mark top-10 cells
    _, img2, d2 = plot_imshow_and_average(ax[:, 1], xtmp, selection=selection, c=colors[0], img_txt_c=img_text_colors[1], img_txt_fs=img_text_fs)

    # sum over selected positions and plot image
    sel = np.zeros(shape=x.shape[1], dtype=bool)
    sel[np.array(preceding)] = True
    xtmp = np.sum(x[:, sel, ...], 1)

    selection, _, _ = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=preceding, seed=12345)  # to mark top-10 cells
    _, img3, d3 = plot_imshow_and_average(ax[:, 2], xtmp, selection=selection, c=colors[1], img_txt_c=img_text_colors[2], img_txt_fs=img_text_fs)

    for a in ax[0, :]:
        a.grid(visible=True, linewidth=0.5)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    return ax, (img1, img2, img3), (d1, d2, d3)


# %%
def make_plot(datadir: str, query: str) -> Tuple[plt.Figure, pd.DataFrame]:

    """
    main plotting routine controling the inputs to and creating the figure

    Parameters:
    ----------
    datadir : str, path to data directory
    query : str, query to plot
        - "colon-colon-p1" (plots the main figure, attending from colon to colon)
    
    Returns:
    -------
    Tuple[plt.Figure, pd.DataFrame]: figure and data

    """

    if query == "colon-colon-p1":
        fn = "attention_weights_gpt2_colon-colon-p1.npz"
        target_id = 13
        query_idx = 45
        suptitle = ""
        n_tokens_per_window = 3

    elif query == "colon-semicolon-p1":
        fn = "attention_weights_gpt2_colon-semicolon-p1.npz"
        target_id = 13
        query_idx = 45
        suptitle = ""
        n_tokens_per_window = 3
    
    elif query == "comma-comma-p1":
        fn = "attention_weights_gpt2_comma-comma-p1.npz"
        target_id = 13
        query_idx = 45  # this is first noun in the list
        suptitle = ""
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ablate-11":
        fn = "attention_weights_gpt2-ablate-10-all_colon-colon-p1-ctxlen1.npz"  # this is 0 indexing
        target_id = 13
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ablated layer 11)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ablate-2":
        fn = "attention_weights_gpt2-ablate-2-all_colon-colon-p1-ctxlen1.npz"  # this is 0 indexing
        target_id = 13
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ablated layer 3)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ctxlen3":
        fn = "attention_weights_gpt2_colon-colon-p1-ctxlen3.npz"  # this is 0 indexing
        target_id = 13
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ctxlen3)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-ctxlen4":
        fn = "attention_weights_gpt2_colon-colon-p1-ctxlen4.npz"  # this is 0 indexing
        target_id = 13
        query_idx = 45
        suptitle = "GPT-2 attention patterns over past context (ctxlen4)"
        n_tokens_per_window = 3

    elif query == "colon-colon-p1-single-token":
        fn = "attention_weights_gpt2_colon-colon-p1.npz"
        target_id = 13
        query_idx = 45
        suptitle = "Copy and previous-token attention heads in GPT-2 (single-token window)"
        n_tokens_per_window = 1

    # attention weights starting at first noun in the list
    elif query == "colon-colon-n1":
        fn = "attention_weights_gpt2_colon-colon-n1.npz"
        target_id = 14
        query_idx = 46
        n_tokens_per_window = 3
        suptitle = "Attention to previous tokens (from first noun in the list)"


    x1, d1 = get_data(os.path.join(datadir, fn))

    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:3]

    fontcfg = get_font_config(pymodule=os.path.basename(__file__), currentfont=plt.rcParams['font.sans-serif'][0])

    # ===== PLOT ===== #

    if query in ["colon-colon-p1", 'colon-semicolon-p1', 'comma-comma-p1', 'colon-colon-n1']:

        fig = plt.figure(figsize=(12.5, 5.5))

        gs = GridSpec(2, 3, height_ratios=[0.3, 1.3], figure=fig)

        ax1 = fig.add_subplot(gs[0, :])

        # first row axes
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
        ax4 = fig.add_subplot(gs[1, 2], sharey=ax2)


        # for the main figure, font can remain smaller (as the figure itself is larger), 
        if query in ["colon-colon-p1"]:
            plot_schematic(ax1, d1, text_fs=fontcfg["schema_text_fs"], annot_fs=fontcfg["schema_annot_fs"], colors=clrs, squeezed=False)
        else:
            plot_schematic(ax1, d1, colors=clrs, squeezed=True)

        axes = np.array([[ax2, ax3, ax4]])

        for a in (ax3, ax4):
            plt.setp(a.get_yticklabels(), visible=False)

    else:

        fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), sharex="all", sharey="row", 
                               gridspec_kw={'height_ratios': [1, 1.8]},
                               )

    # ==== PLOT FIGURE ===== #
    if query in ["colon-colon-p1"] and n_tokens_per_window == 3:
        img_txt_clrs = ["black", "white", "white"]
    elif query in ["colon-colon-p1"] and n_tokens_per_window == 1:
        img_txt_clrs = ["black", "black", "black"]
    else:
        img_txt_clrs = ["black", "black", "white"]

    # ===== PLOTTING FUNCTION ===== #
    ax, imgs, data1 = attn_weights_per_head_layer(ax=axes, 
                                                  x=x1, 
                                                  target_id=target_id, 
                                                  query_id=query_idx,
                                                  colors=clrs, 
                                                  img_text_colors=img_txt_clrs,
                                                  img_text_fs=fontcfg["annotfs"],
                                                  n_tokens_per_window=n_tokens_per_window)

    lfs = fontcfg["labelfs"]
    titlefs = fontcfg["titlefs"]
    #if query in ["comma-comma-p1", "colon-semicolon-p1", "colon-colon-p1-n1"]:
    #    lfs = 22
    #    titlefs = 23

    # ticks and ticklabels
    for a in ax[0, :]:
        a.set_yticks(np.arange(0, 12, 1))
        a.set_yticklabels([str(i) if i%2 != 0 else "" for i in range(1, 13, 1)], fontsize=fontcfg["tickfs"])
        a.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
        a.set_xticks(np.arange(0, 12, 1))
        a.set_xticklabels([str(i) if i%2 != 0 else "" for i in range(1, 13, 1)], fontsize=fontcfg["tickfs"])
 
    # add image plot ticks

    # colorbar
    cax = ax[0, 0].inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(imgs[0], ax=ax[0, 0], cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["", "", ""])

    cax = ax[0, 1].inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(imgs[1], ax=ax[0, 1], cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["", "", ""])

    cax = ax[0, 2].inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(imgs[2], ax=ax[0, 2], cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels([0, 0.5, 1], fontsize=fontcfg["tickfs"])
    cbar.ax.set_ylabel("Attention weight", fontsize=fontcfg["labelfs"], fontweight="semibold")
    cbar.ax.tick_params(labelsize=lfs)

    ax[0, 0].set_ylabel("Layer", fontsize=fontcfg["labelfs"], fontweight="semibold")
    fig.supxlabel("Head", fontsize=fontcfg["labelfs"], fontweight="semibold")
    fig.suptitle(suptitle, fontsize=titlefs)
    fig.tight_layout()

    # format data in a csv
    datarec_ax0 = {f"layer-{i+1}": v for i, v in enumerate(data1[0][0])}   # to nouns
    datarec_ax1 = {f"layer-{i+1}": v for i, v in enumerate(data1[1][0])}   # to nouns
    data = pd.DataFrame.from_dict([datarec_ax0, datarec_ax1]).T
    data.columns = ['distant', 'recent']


    return fig, data

# %%
def save_png_pdf(fig, savename: str):

    savefn = os.path.join(savename + ".png")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, dpi=300, format="png")

    savefn = os.path.join(savename + ".pdf")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, format="pdf", transparent=True, bbox_inches="tight")

    return 0

# %%
def main(input_args=None):

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, choices=["all", "main_fig", "control_fig", "single_token_fig", "first_noun_fig"])
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    logger.info(f"Using {args.datadir} as data directory")
    datadir = args.datadir

    if args.which is None:
        raise ValueError("Please specify which figures to make by providing '--which' argument")

    if args.which == "all":
        
        main_fig = True
        control_fig = True
        single_token_fig = True
        first_noun_fig = True
        ctxlen3_fig = True
        ctxlen4_fig = True

    else:

        main_fig = True if args.which == "main_fig" else False
        control_fig = True if args.which == "control_fig" else False
        single_token_fig = True if args.which == "single_token_fig" else False
        first_noun_fig = True if args.which == "first_noun_fig" else False
        ctxlen3_fig = True if args.which == "ctxlen3_fig" else False
        ctxlen4_fig = True if args.which == "ctxlen4_fig" else False


    set_manuscript_style()

    if main_fig:

        fig1, data = make_plot(datadir=datadir, query="colon-colon-p1")

        if args.savedir:
            save_png_pdf(fig1, os.path.join(args.savedir, "gpt2_attn_colon-colon-p1_v2"))

            fn = os.path.join(args.savedir, "gpt2_attn_colon-colon-p1_v2" + ".csv")
            logging.info(f"Saving {fn}")
            data.to_csv(fn, sep='\t')

    # ===== CONTROL FIGURES: SWAP QUERY TOKENS ==== #
    if control_fig:

        fig2, _, data = make_plot(datadir=datadir, query="colon-semicolon-p1")

        if args.savedir:

            savename = "gpt2_attn_colon-semicolon-p1"
            save_png_pdf(fig2, os.path.join(args.savedir, savename))

            fn = os.path.join(args.savedir, savename + ".csv")
            logging.info(f"Saving {fn}")
            data.to_csv(fn, sep='\t')


        fig3, _, data = make_plot(datadir=datadir, query="comma-comma-p1")
        if args.savedir:

            savename = "gpt2_attn_comma-comma-p1"
            save_png_pdf(fig3, os.path.join(args.savedir, savename))
            
            fn = os.path.join(args.savedir, savename + ".csv")
            logging.info(f"Saving {fn}")
            data.to_csv(fn, sep='\t')

    # single token plot
    if single_token_fig:
        fig4, fig4_, data = make_plot(datadir=datadir, query="colon-colon-p1-single-token")
        if args.savedir:
            save_png_pdf(fig4, os.path.join(args.savedir, "gpt2_attn_colon-colon-p1-single-token"))


    if first_noun_fig:

        fig5, fig5_, data = make_plot(datadir=datadir, query="colon-colon-n1")
        if args.savedir:
            save_png_pdf(fig5, os.path.join(args.savedir, "gpt2_attn_colon-colon-n1"))
            save_png_pdf(fig5_, os.path.join(args.savedir, "gpt2_attn_colon-colon-n1_control"))

            fn = os.path.join(args.savedir, "gpt2_attn_colon-colon-n1" + ".csv")
            logging.info(f"Saving {fn}")
            data.to_csv(fn, sep='\t')

        # ===== CONTROL FIGURES: ABLATED MODEL ==== #
        #fig5, fig5_, data = make_plot(datadir=datadir, query="colon-colon-p1-ablate-11")
        #if args.savedir:
        #    save_png_pdf(fig5, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1"))
        #    save_png_pdf(fig5_, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1_control"))

        #    fn = os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1")
        #    logging.info(f"Saving {fn}")
        #    data.to_csv(fn, sep='\t')

        #fig6, fig6_, data = make_plot(datadir=datadir, query="colon-colon-p1-ablate-2")
        #if args.savedir:
        #    save_png_pdf(fig6, os.path.join(args.savedir, "gpt2-ablate-2_attn_colon-colon-p1"))
        #    save_png_pdf(fig6_, os.path.join(args.savedir, "gpt2-ablate-2_attn_colon-colon-p1_control"))

        #    fn = os.path.join(args.savedir, "gpt2-ablate-2_attn_colon-colon-p1")
        #    logging.info(f"Saving {fn}")
        #    data.to_csv(fn, sep='\t')


        # ===== CONTROL FIGURES: LONG CONTEXT ==== #
        #fig7, fig7_ = make_plot(datadir=datadir, query="colon-colon-p1-ctxlen3")
        #if args.savedir:
        #    save_png_pdf(fig7, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1-ctxlen3"))
        #    save_png_pdf(fig7_, os.path.join(args.savedir, "gpt2-ablate-11_attn_colon-colon-p1-ctxlen3_control"))
        
        #fig8, fig8_ = make_plot(datadir=datadir, query="colon-colon-p1-ctxlen4")
        #if args.savedir:
        #    save_png_pdf(fig8, os.path.join(args.savedir, "gpt2-ablate-0_attn_colon-colon-p1-ctxlen4"))
        #    save_png_pdf(fig8_, os.path.join(args.savedir, "gpt2-ablate-0_attn_colon-colon-p1-ctxlen4_control"))

        plt.show()

    #plt.close("all")

    #with plt.style.context("seaborn-whitegrid"):

    #    fig1, fig2 = generate_plot(datadir=datadir, query="colon-semicolon-p1")

# %%
if __name__ == "__main__":

    main()

