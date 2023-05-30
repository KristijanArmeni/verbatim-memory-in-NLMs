import os, sys
sys.path.append(os.environ["PROJ_ROOT"])

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from attn_weights_per_layer_figure import get_data, save_png_pdf
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
    im = ax[1].imshow(x[i, 0:query_id, :, l].T, aspect='auto', cmap="Greys")

    labelfs = 16  # fontsize of axis and tick labels

    ax[1].set_xlabel('Input sequence', fontsize=labelfs)
    ax[1].set_ylabel(f'Head\n(Layer {layer+1})', fontsize=labelfs)
    ax[1].set_xticks(ticks=np.arange(query_id))

    # line plot
    m = np.mean(x[i, 0:query_id, :, l], axis=1)

    ax[0].plot(np.arange(query_id), m, 'o--', markersize=7, mec='white')

    ax[0].set_ylabel('Avg. attention\nweight', fontsize=labelfs)
    ax[0].tick_params(labelsize=labelfs)
    ax[0].grid(visible=True, linewidth=0.5)
    
    # despine axis 1
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    
    ylabels = [s.strip("Ġ") for s in d['tokens'][i][0:query_id]]
    ax[1].set_xticklabels(labels=ylabels, ha='right', rotation=55, fontsize=labelfs)
    ax[1].set_yticks(np.arange(0, 12, 2))
    ax[1].set_yticklabels(np.arange(1, 12, 2), fontsize=labelfs)

    # place rectangles
    rect = patches.Rectangle((query_id-1.5, -0.5), 1, 12, linewidth=2, ec='tab:red', facecolor='none', zorder=2)
    cue1 = patches.Rectangle((12.5, -0.5), 1, 12, linewidth=2, ec='tab:red', facecolor='none', zorder=2)
    early_window_rect = patches.Rectangle((13.5, -0.5), 5, 12, linewidth=2, edgecolor='tab:blue', facecolor='none', zorder=2)
    preceding_window_rect = patches.Rectangle((query_id-4.5, -0.5), 3, 12, linewidth=2, edgecolor='tab:orange', facecolor='none', zorder=1)
    for patch in (cue1, rect, early_window_rect):
        ax[1].add_patch(patch)
    
    patch_height = 0.29
    cue2 = patches.Rectangle((query_id-1.5, 0), 1, patch_height, linewidth=2, edgecolor='tab:red', facecolor='tab:red', alpha=0.2, zorder=2)
    cue1 = patches.Rectangle((12.5, 0), 1, patch_height, linewidth=2, ec='tab:red', facecolor='tab:red', alpha=0.2, zorder=2)
    early_window_rect = patches.Rectangle((13.5, 0), 5, patch_height, linewidth=2, edgecolor='tab:blue', facecolor='tab:blue', alpha=0.2, zorder=2)
    preceding_window_rect = patches.Rectangle((query_id-4.5, 0), 3, patch_height, linewidth=2, edgecolor='tab:orange', facecolor='tab:orange', alpha=0.2, zorder=2)
    for patch in (cue1, early_window_rect, cue2):
        ax[0].add_patch(patch)

    ax[0].text(x=12.5, y=0.32, s="$c_t$", fontsize=16, fontweight="bold")
    ax[0].text(x=14, y=0.32, s="$n_{t+1}$", fontsize=16, fontweight="bold")
    ax[0].text(x=query_id-1.5, y=0.32, s="$c^{\prime}_{t+k}$", fontsize=16, fontweight="bold")

    ax[0].annotate("", xy=(16, 0.33), xytext=(query_id-1.5, 0.33), 
                  arrowprops={"arrowstyle": "->", 'connectionstyle': "arc3,rad=0.1"})

    ax[0].text(x=22, y=0.33, s="attention towards tokens in repeated sequences", fontsize=14)

    cax = ax[1].inset_axes([1.04, 0.15, 0.01, 0.7])
    cbar = plt.colorbar(im, ax=ax[1], cax=cax, shrink=0.5, anchor=(0.5, 1))

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel("Attention\nweight", rotation=90, fontsize=labelfs)
    cbar.ax.tick_params(labelsize=labelfs)


    return ax


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
        plt.show()

        if args.savedir:
            save_png_pdf(fig, savename=os.path.join(args.savedir, f"attn_weights_example_{lay}-{seq}"))


    return 0


if __name__ == "__main__":


    main()