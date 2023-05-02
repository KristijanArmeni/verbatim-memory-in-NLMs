import os, sys
sys.path.append(os.environ["PROJ_ROOT"])

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import median_abs_deviation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib as mpl
from src.wm_suite.viz.func import filter_and_aggregate
from attn_weights_per_layer_figure import get_data, save_png_pdf
import logging
from typing import List, Dict, Tuple
from itertools import product

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

    #ax[1].set_xlabel('Token', fontsize=labelfs)
    ax[1].set_ylabel('Head', fontsize=labelfs)
    ax[1].set_xticks(ticks=np.arange(query_id))

    # line plot
    m = np.mean(x[i, 0:query_id, :, l], axis=1)
    #se = sem(x[i, 0:query_id, :, l], axis=1)

    ax[0].plot(np.arange(query_id), m, 'o--', markersize=7, mec='white')
    #ax[0].fill_between(np.arange(query_id), y1=m+se, y2=m-se, alpha=0.3)
    ax[0].set_ylabel('Avg.\nattn. w.', fontsize=labelfs)
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
    for patch in (rect, early_window_rect, preceding_window_rect):
        ax[1].add_patch(patch)
 
    rect = patches.Rectangle((query_id-1.5, 0), 1, 0.25, linewidth=2, edgecolor='tab:red', facecolor='tab:red', alpha=0.2, zorder=2)
    early_window_rect = patches.Rectangle((13.5, 0), 5, 0.25, linewidth=2, edgecolor='tab:blue', facecolor='tab:blue', alpha=0.2, zorder=2)
    preceding_window_rect = patches.Rectangle((query_id-4.5, 0), 3, 0.25, linewidth=2, edgecolor='tab:orange', facecolor='tab:orange', alpha=0.2, zorder=2)
    for patch in (rect, early_window_rect, preceding_window_rect):
        ax[0].add_patch(patch)


    cax = ax[1].inset_axes([1.04, 0.15, 0.01, 0.7])
    cbar = plt.colorbar(im, ax=ax[1], cax=cax, shrink=0.5, anchor=(0.5, 1))

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel("Attention\nweight", rotation=90, fontsize=labelfs)
    cbar.ax.tick_params(labelsize=labelfs)
    
    return ax


def generate_plot(datadir, layer, sequence):

    fn = "attention_weights_gpt2_colon-colon-p1.npz"
    query_idx = 46
    suptitle = f"GPT-2 attention weights over context (layer {layer+1})"

    # load the data
    x, d = get_data(os.path.join(datadir, fn))

    # make figure
    fig, ax = plt.subplots(2, 1, figsize=(14, 4.5), sharex='col', gridspec_kw={'height_ratios': [0.9, 1.8]})

    # plot
    make_example_plot(ax, x=x, d=d, query_id=query_idx, layer=layer, sequence=sequence)

    plt.suptitle(suptitle, fontsize=18)
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


    with plt.style.context("seaborn-ticks"):
        lay, seq = 2, 5
        fig = generate_plot(datadir=args.datadir, layer=lay, sequence=seq)

        if args.savedir:
            save_png_pdf(fig, savename=os.path.join(args.savedir, f"attn_weights_example_{lay}-{seq}"))

        lay, seq = 10, 5
        fig = generate_plot(datadir=args.datadir, layer=lay, sequence=seq)
        plt.show()

        if args.savedir:
            save_png_pdf(fig, savename=os.path.join(args.savedir, f"attn_weights_example_{lay}-{seq}"))


    return 0


if __name__ == "__main__":


    main()