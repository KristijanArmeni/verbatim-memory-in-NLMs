# %%

import os
import json
from glob import glob
from typing import List, Tuple

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from paths import PATHS
from src.wm_suite.viz.utils import clrs


# %%
fs = glob(os.path.join(PATHS.search, "*.json"))

# %%


# load the search results
def load_json(filenames: List) -> Tuple:
    y, y_ci = [], []
    x1, x1_ci = [], []
    x2, x2_ci = [], []
    x = []

    for f in filenames:
        with open(f, "r") as fh:
            d = json.load(fh)
            y.append(d["rs"]["scores"])
            y_ci.append(d["rs"]["ci"])
            x1.append(d["x1"]["scores"])
            x1_ci.append(d["x1"]["ci"])
            x2.append(d["x2"]["scores"])
            x2_ci.append(d["x2"]["ci"])
            x.append([len(e) for e in d["best_labels"]])

    yarr = np.vstack(y)
    y_ciarr = np.stack(y_ci)
    x1arr = np.vstack(x1)
    x1_ciarr = np.stack(x1_ci)
    x2arr = np.vstack(x2)
    x2_ciarr = np.stack(x2_ci)
    xarr = np.vstack(x)

    return (yarr, y_ciarr), (xarr,), (x1arr, x1_ciarr), (x2arr, x2_ciarr)


# %%
fn = os.path.join(PATHS.search, "gpt2_sce1_llen3_plen1_repeat_random.json")
with open(fn, "r") as fh:
    y_unab = json.load(fh)

# %%


def plot_search(xarr, yarr, y_ciarr):
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    ax.plot(xarr[0, :], yarr[0, :], "o--", label="Matching", color=clrs.green)
    ax.plot(xarr[1, :], yarr[1, :], "s--", label="Postmatch", color=clrs.blue)
    ax.plot(xarr[2, :], yarr[2, :], "^--", label="Recent-tokens", color=clrs.orange)

    ax.vlines(
        xarr[0, np.argmax(yarr[0, :])],
        0,
        np.max(yarr[0, :]),
        linestyle="--",
        color=clrs.green,
    )
    ax.vlines(
        xarr[1, np.argmax(yarr[1, :])],
        0,
        np.max(yarr[1, :]),
        linestyle="--",
        color=clrs.blue,
    )
    ax.vlines(
        xarr[2, np.argmax(yarr[2, :])],
        0,
        np.max(yarr[2, :]),
        linestyle="--",
        color=clrs.orange,
    )

    ax.fill_between(
        xarr[0, :],
        y1=y_ciarr[0, :, 0],
        y2=y_ciarr[0, :, 1],
        color=clrs.green,
        alpha=0.3,
    )
    ax.fill_between(
        xarr[1, :], y1=y_ciarr[1, :, 0], y2=y_ciarr[1, :, 1], color=clrs.blue, alpha=0.3
    )
    ax.fill_between(
        xarr[2, :],
        y1=y_ciarr[2, :, 0],
        y2=y_ciarr[2, :, 1],
        color=clrs.orange,
        alpha=0.3,
    )

    ax.set_xlabel("Number of heads ablated (combination with largest effect)")
    ax.set_ylabel("Median repeat surprisal (%)")

    ax.set_xticks(xarr[0, :])
    ax.set_xticklabels(xarr[0, :])

    ax.set_title("Greedy search per head type")
    ax.legend(title="Head type")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

    plt.tight_layout()

    return fig


# fig.savefig(os.path.join(PATHS.root, "fig", "ablation", "topk", "greedy_search_per_head_type.png"), dpi=300, format="png")


# %%
def plot_search_x1x2(xarr, y1tup, y2tup, y3tup, y4tup):
    x1arr, x1_ciarr = y1tup[0], y1tup[1]
    x2arr, x2_ciarr = y2tup[0], y2tup[1]
    x1ctrl, x1ctrl_ci = y3tup[0], y3tup[1]
    x2ctrl, x2ctrl_ci = y4tup[0], y4tup[1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    ax[0].plot(xarr[0, :], x1arr[0, :], "o--", label="Matching", color=clrs.green)
    ax[0].plot(xarr[1, :], x1arr[1, :], "s--", label="Postmatch", color=clrs.blue)
    ax[0].plot(xarr[2, :], x1arr[2, :], "^--", label="Recent-tokens", color=clrs.orange)

    ax[0].fill_between(
        xarr[0, :],
        y1=x1_ciarr[0, :, 0],
        y2=x1_ciarr[0, :, 1],
        color=clrs.green,
        alpha=0.3,
    )
    ax[0].fill_between(
        xarr[1, :],
        y1=x1_ciarr[1, :, 0],
        y2=x1_ciarr[1, :, 1],
        color=clrs.blue,
        alpha=0.3,
    )
    ax[0].fill_between(
        xarr[2, :],
        y1=x1_ciarr[2, :, 0],
        y2=x1_ciarr[2, :, 1],
        color=clrs.orange,
        alpha=0.3,
    )

    ax[1].plot(xarr[0, :], x2arr[0, :], "o--", label="Matching", color=clrs.green)
    ax[1].plot(xarr[1, :], x2arr[1, :], "s--", label="Postmatch", color=clrs.blue)
    ax[1].plot(xarr[2, :], x2arr[2, :], "^--", label="Recent-tokens", color=clrs.orange)

    ax[1].fill_between(
        xarr[0, :],
        y1=x2_ciarr[0, :, 0],
        y2=x2_ciarr[0, :, 1],
        color=clrs.green,
        alpha=0.3,
    )
    ax[1].fill_between(
        xarr[1, :],
        y1=x2_ciarr[1, :, 0],
        y2=x2_ciarr[1, :, 1],
        color=clrs.blue,
        alpha=0.3,
    )
    ax[1].fill_between(
        xarr[2, :],
        y1=x2_ciarr[2, :, 0],
        y2=x2_ciarr[2, :, 1],
        color=clrs.orange,
        alpha=0.3,
    )

    unabl_x = np.arange(2.5, 21.5)
    ax[0].hlines(x1ctrl, 2.5, 20.5, linestyle="--", color="black", label="Unablated")
    ax[0].fill_between(
        unabl_x, y1=x1ctrl_ci[0], y2=x1ctrl_ci[1], color="black", alpha=0.3
    )

    # plot control surprisal on x2
    ax[1].hlines(x2ctrl, 2.5, 20.5, linestyle="--", color="black", label="Unablated")
    ax[1].fill_between(
        unabl_x, y1=x2ctrl_ci[0], y2=x2ctrl_ci[1], color="black", alpha=0.3
    )

    fig.supxlabel(
        "Number of heads ablated (combination with largest effect on repeat surprisal)"
    )
    ax[0].set_ylabel("Median surprisal (95% CI)")

    for a in ax:
        a.set_xticks(xarr[0, :])
        a.set_xticklabels(xarr[0, :])
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.yaxis.set_minor_locator(AutoMinorLocator(2))
        a.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

    fig.suptitle("Effect of ablations on raw surprisal scores (list-initial token)")
    ax[0].set_title("First list")
    ax[1].set_title("Second list")
    ax[0].legend(title="Head type")

    plt.tight_layout()


# fig.savefig(os.path.join(PATHS.root, "fig", "ablation", "topk", "greedy_search_per_head_type_surp.png"), dpi=300, format="png")


# %%

y, x, x1, x2 = load_json(filenames=fs[3::])
ctrl_tup_x1 = (y_unab["x1"]["median"], y_unab["x1"]["ci95"])
ctrl_tup_x2 = (y_unab["x2"]["median"], y_unab["x2"]["ci95"])

y_n1, x_n1, x1_n1, x2_n1 = load_json(filenames=fs[0:1] + fs[2:3] + fs[1:2])
ctrl_tup_x1 = (y_unab["x1"]["median"], y_unab["x1"]["ci95"])
ctrl_tup_x2 = (y_unab["x2"]["median"], y_unab["x2"]["ci95"])


# %%
fig = plot_search(xarr=x[0], yarr=y[0], y_ciarr=y[1])
plt.title("Greedy search per head type (query = `:`)")
plt.show()

fig = plot_search_x1x2(
    xarr=x[0], y1tup=x1, y2tup=x2, y3tup=ctrl_tup_x1, y4tup=ctrl_tup_x2
)
plt.show()

# %%

fig = plot_search(xarr=x_n1[0], yarr=y_n1[0], y_ciarr=y_n1[1])
plt.show()

fig = plot_search_x1x2(
    xarr=x_n1[0], y1tup=x1_n1, y2tup=x2_n1, y3tup=ctrl_tup_x1, y4tup=ctrl_tup_x2
)
plt.show()

# %%
