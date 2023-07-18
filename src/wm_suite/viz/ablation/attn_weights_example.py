import os, sys
sys.path.append(os.environ["PROJ_ROOT"])

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from attn_weights_figure import get_data, save_png_pdf
from src.wm_suite.viz.func import set_manuscript_style
import logging
from typing import List, Dict, Tuple


logging.basicConfig(level=logging.INFO, format="%(message)s")


def make_example_plot(ax: np.ndarray, x: np.ndarray, d: Dict, query_id: int, layer: int, sequence:int):
    """
    wrapper to plot attention weights as lineplots per layer
    
    Parameters:
    ---------
    x : np.array (sequence, tokens, heads, layer)
    d : dict with fields 'data', 'tokens'
    layer : int
        integer indicating which layer to plot (0-indexing)
    sequence : int
        integer indicating which sequence to plot
    
    Returns:
    fig : matplotlib figure object
    ax : matplotlib axes object
    """

    i = sequence
    l = layer
    im = ax[1].imshow(x[i, 0:query_id, :, l].T, aspect='auto', vmin=0, vmax=1, cmap="Greys")

    labelfs = 16  # fontsize of axis and tick labels

    ax[1].set_xlabel('Input sequence', fontsize=labelfs)
    ax[1].set_ylabel(f'Head\n(Layer {layer+1})', fontsize=labelfs)
    ax[1].set_xticks(ticks=np.arange(query_id))

    # line plot
    m = np.mean(x[i, 0:query_id, :, l], axis=1)

    ax[0].plot(np.arange(query_id), m, 'o--', markersize=8, mec='white')
    
    ax[0].set_yticks([0, 0.1, 0.2, 0.3])
    ax[0].set_yticklabels([0, 0.1, 0.2, 0.3])
    ax[0].tick_params(labelsize=labelfs-1)

    ax[0].set_ylabel('Avg. attention\nweight', fontsize=labelfs)

    ax[0].grid(visible=True, linewidth=0.5)
    
    # despine axis 1
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    
    ylabels = [s.strip("Ġ") for s in d['tokens'][i][0:query_id]]
    ylabels = [s.replace("<|endoftext|>", "<eos>") for s in ylabels]
    ax[1].set_xticklabels(labels=ylabels, ha='center', rotation=90, fontsize=labelfs)
    ax[1].set_yticks(np.arange(0, 12, 2))
    ax[1].set_yticklabels(np.arange(1, 12, 2), fontsize=labelfs)

    # place rectangles
    rect = patches.Rectangle((query_id-1.5, -0.5), 1, 12, linewidth=2, ec='tab:red', facecolor='none', zorder=2)
    cue1 = patches.Rectangle((12.5, -0.5), 1, 12, linewidth=2, ec='tab:red', facecolor='none', zorder=2)
    early_window_rect = patches.Rectangle((13.5, -0.5), 5, 12, linewidth=2, edgecolor='tab:blue', facecolor='none', zorder=2)
    preceding_window_rect = patches.Rectangle((query_id-4.5, -0.5), 3, 12, linewidth=2, edgecolor='tab:orange', facecolor='none', zorder=1)
    #for patch in (cue1, rect, early_window_rect):
    #    ax[1].add_patch(patch)
    
    patch_height = 0.3
    cue2 = patches.Rectangle((query_id-1.5, 0), 1, patch_height, linewidth=2, facecolor='tab:red', alpha=0.2, zorder=2)
    cue1 = patches.Rectangle((12.5, 0), 1, patch_height, linewidth=2, facecolor='tab:red', alpha=0.2, zorder=2)
    early_window_rect = patches.Rectangle((13.5, 0), 5, patch_height, linewidth=2, facecolor='tab:blue', alpha=0.2, zorder=2)
    preceding_window_rect = patches.Rectangle((query_id-4.5, 0), 3, patch_height, linewidth=2, facecolor='tab:orange', alpha=0.2, zorder=2)
    for patch in (cue1, early_window_rect, cue2):
        ax[0].add_patch(patch)

    ax[0].text(x=12.5, y=0.32, s="$c_t$", fontsize=16, fontweight="bold")
    ax[0].text(x=14, y=0.32, s="$n_{t+1}$", fontsize=16, fontweight="bold")
    ax[0].text(x=query_id-1.5, y=0.32, s="$c^{\prime}_{t+k}$", fontsize=16, fontweight="bold")

    ax[0].annotate("", xy=(16, 0.33), xytext=(query_id-1.5, 0.33), 
                  arrowprops={"arrowstyle": "->", 'connectionstyle': "arc3,rad=0.1"})

    ax[0].text(x=22, y=0.33, s="Attention towards tokens in repeated sequences", fontsize=14)

    cax = ax[1].inset_axes([1.02, 0, 0.01, 1])
    cbar = plt.colorbar(im, ax=ax[1], cax=cax, shrink=0.5, anchor=(0.5, 1))

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel("Attention weight", rotation=90, fontsize=labelfs)
    cbar.ax.tick_params(labelsize=labelfs)


    return ax


def replace_nouns(x, replacements):
   d = {"patience": replacements[0], "notion": replacements[1], "movie": replacements[2]}
   return [d[e] if e in d.keys() else e for e in x]


def make_example_plot2(datadir):

    x, d = get_data(os.path.join(datadir, "attention_weights_gpt2_colon-colon-p1.npz"))

    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex="all")
    
    t_start=0
    t_end=46

    xticklabels = replace_nouns([e.strip("Ġ") for e in d["tokens"][0][t_start:t_end]], ["N1", "N2", "N2"])

    layers = [0, 4, 10]

    data = np.mean(x, axis=0) # average across sequences

    im1 = ax[0].imshow(data[:, :, layers[0]][t_start:t_end, ...].T, vmin=0, vmax=1, cmap=plt.cm.Blues)
    im2 = ax[1].imshow(data[:, :, layers[1]][t_start:t_end, ...].T, vmin=0, vmax=1, cmap=plt.cm.Blues)
    im3 = ax[2].imshow(data[:, :, layers[2]][t_start:t_end, ...].T, vmin=0, vmax=1, cmap=plt.cm.Blues)
    
    cax1 = ax[0].inset_axes([1.02, 0, 0.02, 1])
    cax2 = ax[1].inset_axes([1.02, 0, 0.02, 1])
    cax3 = ax[2].inset_axes([1.02, 0, 0.02, 1])
    
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)

    lfs = 12

    cbar1.ax.set_ylabel("Avg. attention weight")
    cbar2.ax.set_ylabel("Avg. attention weight")
    cbar3.ax.set_ylabel("Avg. attention weight")

    ax[2].set_xlabel("Token", fontsize=lfs)
    ax[2].set_xticks(np.arange(len(xticklabels)))
    ax[2].set_xticklabels(xticklabels, rotation=90, fontsize=10)

    for i, a in enumerate(ax):
        a.set_title(f"Layer {layers[i]+1}")

    for a in ax:
        a.set_yticks(np.arange(0, 12, 1))
        a.set_yticklabels(np.arange(0, 12, 1)+1)

    fig.supylabel("Attention head", fontsize=lfs)
    plt.suptitle("Average attention weights across sequences (N = 230)")

    for a in ax:
        a.grid(visible=True, linewidth=0.5)
    plt.tight_layout()

    return fig


def generate_plot(datadir, layer, sequence):

    fn = "attention_weights_gpt2_colon-colon-p1.npz"
    query_idx = 46
    suptitle = f"Transformer attention for short-term memory"

    # load the data
    x, d = get_data(os.path.join(datadir, fn))

    # make figure
    fig, ax = plt.subplots(2, 1, figsize=(14, 5.5), sharex='col', gridspec_kw={'height_ratios': [0.9, 1.8]})

    # plot
    make_example_plot(ax, x=x, d=d, query_id=query_idx, layer=layer, sequence=sequence)

    plt.suptitle(suptitle, fontsize=20, fontweight="bold")
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

    with plt.style.context("seaborn-ticks"):
        #lay, seq = 2, 5
        #fig = generate_plot(datadir=args.datadir, layer=lay, sequence=seq)

        #if args.savedir:
        #    save_png_pdf(fig, savename=os.path.join(args.savedir, f"attn_weights_example_{lay}-{seq}"))

        lay, seq = 10, 5
        fig = generate_plot(datadir=args.datadir, layer=lay, sequence=seq)

        if args.savedir:
            save_png_pdf(fig, savename=os.path.join(args.savedir, f"attn_weights_example_{lay}-{seq}"))

        fig = make_example_plot2(args.datadir)

        if args.savedir:
            save_png_pdf(fig, savename=os.path.join(args.savedir, f"attn_weights_avg_layer-1-5-11"))

        plt.show()

    return 0


if __name__ == "__main__":


    main()