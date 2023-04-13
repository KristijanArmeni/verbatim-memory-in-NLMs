import os, json
import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import median_abs_deviation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from viz.func import filter_and_aggregate
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
    x : np.ndarray, shape = (layers, timesteps, samples, heads)
        the data array containing weihgs
    axis: tuple
        axis over which to average

    """
    m = np.mean(x, axis=axis)
    se = sem(np.mean(x, axis=0), axis=0)
    
    return m, se


def plot_attention(ax: mpl.axes.Axes, x: np.ndarray, token_ids: np.ndarray, labels: List):
    """
    Parameters:
    ----------
    ax: mpl.Axes
        axis (single dimensional) onto which to plot attention weigths
    x: np.ndarray, shape = (layers, timesteps, sequences, heads)

    """
    for t, l in zip(token_ids, labels):

        # average acrross heads and across sequences
        m, se = get_mean_se(x[:, t, :, :], axis=(0, 1))

        ax.plot(m, '--o', label=l)
        ax.fill_between(np.arange(m.shape[-1]), y1=m+se, y2=m-se, alpha=0.2)

    ax.set_ylabel('average attention weight\n(across heads & sequences)')
    ax.set_xlabel('layer')
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(np.arange(12)+1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def attn_amplitude_plot(gs: GridSpec, x: np.ndarray, d: Dict, target_id=13, query_id=45, seed=12345):

    i = 2 #
    xs = np.arange(12)
    ylabels = [s.strip('Ġ') for s in d['tokens'][0]]
    
    ids1 = (target_id, target_id+1, target_id+3, target_id+5)
    labels = [f"':' (first list) [{ids1[0]+1}]", 
              f"first noun [{ids1[1]+1}]", 
              f"second noun [{ids1[2]+1}]", 
              f"third noun [{ids1[3]+1}]"]

    ax1, ax2, ax3 = plt.subplot(gs[0, :]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
    for a in (ax1, ax2, ax3):
        a.set_prop_cycle('color', plt.cm.YlGnBu(np.linspace(0.4, 0.9, len(labels))))
        
    plot_attention(ax1, x, ids1, labels)
    query_ids = (query_id-1, query_id-2, query_id-3)
    labs = (f"{ylabels[q]} (t-{i+1}) [{q}]" for i, q in enumerate(query_ids))
    plot_attention(ax2, x, query_ids, labs)

    np.random.seed(seed)
    ints = np.random.choice(np.arange(20, 35), 3)
    labs = [f"{ylabels[i]} [{i+1}]" for i in ints]
    plot_attention(ax3, x, ints, labs)

    ax1.set_title("To first list")
    ax2.set_title("To immediately preceeding tokens")
    ax3.set_title("To random intermediate tokens (control)")

    ax3.set_ylabel("")
    ax3.set_yticklabels("")
    ax3.spines['left'].set_visible(False)
    
    return ax1, ax2, ax3
    

def generate_plot(datadir:str, query: str):

    if query == "colon":
        fn = "gpt2_attn.npz"

    x1, d1 = get_data(os.path.join(datadir, fn))

    gs = GridSpec(2, 2)

    fig = plt.figure(figsize = (11, 7))

    ax1, ax2, ax3 = attn_amplitude_plot(gs, x1, d1)

    for a in (ax1, ax2, ax3):
        a.set_ylim(0, 0.3)
        a.legend(title="Attention to", fontsize=12)

    plt.suptitle("GPT2 attention per layer ({query} as query token)", fontsize=18)
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

    datadir = check_datadir(args.datadir)

    fig = generate_plot(datadir=datadir, query = "colon")

    savefn = os.path.join(args.savedir, "gpt2_attn_query-colon.png")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, dpi=300)

    savefn = os.path.join(args.savedir, "gpt2_attn_query-colon.pdf")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, transparent=True, bbox_inches="tight")


if __name__ == "__main__":

    main()