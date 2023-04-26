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



def generate_plot(datadir):

    #load data
    x, d = get_data(os.path.join(datadir, "gpt2_attn.npz"))
    
    labels = [s.strip("Ġ") for s in d['tokens'][0][0:20]]
    yvalues = np.zeros(len(labels))
    yvalues[0:10] = 0.4
    yvalues[10::] = 0.2
    xvalues = np.tile(np.arange(10), 2)
    

    fig, ax = plt.subplots(figsize=(10, 4))

    spacing = 4

    #ax.set_xlim([0, len(labels)*spacing])
    #ax.set_axis_off()

    for i, tk in enumerate(labels):

        ax.text(xvalues[i]*spacing, yvalues[i], tk, color="black", fontsize=14, ha='left')

    ax.set_xticks(np.arange(0, 40, spacing))

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