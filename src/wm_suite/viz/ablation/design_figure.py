import os, sys
sys.path.append(os.environ["PROJ_ROOT"])

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import median_abs_deviation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from src.wm_suite.viz.func import filter_and_aggregate
from attn_weights_per_layer_figure import get_data
import logging
from typing import List, Dict, Tuple
from itertools import product



def plot_schematic(d, ax):

    #load data
    _, d = get_data(os.path.join(datadir, "gpt2_attn.npz"))
    
    first_list = " ".join([s.strip("Ġ") for s in d['tokens'][0][13:20]])
    ctx = " ".join([s.strip("Ġ") for s in d['tokens'][0][20:35]])
    second_list = " ".join([s.strip("Ġ") for s in d['tokens'][0][-11:-1]])

    #print(prefix)
    #print(ctx)
    #print(second_list)

    labels = ["PREFIX", "... " + first_list, "INTERVENING TEXT", "... " + second_list]
    xpos = [1, 4, 15, 22.5]
    ypos = [0.4, 0.4, 0.4, 0.4]
    xarrow_from = [27.9, 27.9]
    xarrow_to = [9, 25]
    fractions = [0.07, 0.2]

    fig, ax = plt.subplots(figsize=(15, 3))

    spacing = 4

    #ax.set_xlim([0, len(labels)*spacing])
    #ax.set_axis_off()

    for i, tk in enumerate(labels):

        if i in [0, 2]:
            ax.text(xpos[i], ypos[i], tk, color="black", fontsize=14, ha='left',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

        else:    
            ax.text(xpos[i], ypos[i], tk, color="black", fontsize=14, ha='left')

    for i in range(len(xarrow_from)):

            ax.annotate('', 
                        xy=(xarrow_to[i], 0.5),
                        xytext=(xarrow_from[i], 0.5), 
                        va='center', 
                        ha='center',
                        fontsize=14,
                        arrowprops={'arrowstyle': '->', 'ls':'dashed', 'connectionstyle': f'bar,fraction={fractions[i]}'})


    ax.set_xticks(np.arange(0, 40))

    return fig


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

    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args()

    fig = generate_plot(datadir = args.datadir)
    plt.show()


    if args.savedir:

        fn = os.path.join(args.savedir, "design_fig")
        save_png_pdf(fig, savename=fn)


    return 0


if __name__ == "__main__":

    main()