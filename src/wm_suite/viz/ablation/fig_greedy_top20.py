# %%

import os
import json
from glob import glob
from typing import Any, List, Tuple, Dict

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from paths import PATHS
from src.wm_suite.viz.utils import clrs
from src.wm_suite.wm_ablation import from_labels_to_dict

# %% define all files

filenames = [os.path.join(PATHS.search, "greedy_search_matching.json"),
            os.path.join(PATHS.search, "greedy_search_matching_n1.json"),
            os.path.join(PATHS.search, "greedy_search_matching_n1_s2.json"),
            os.path.join(PATHS.search, "greedy_search_postmatch.json"),
            os.path.join(PATHS.search, "greedy_search_postmatch_n1.json"),
            os.path.join(PATHS.search, "greedy_search_postmatch_n1_s2.json"),
            os.path.join(PATHS.search, "greedy_search_recent.json"),
            os.path.join(PATHS.search, "greedy_search_recent_n1.json"),
            os.path.join(PATHS.search, "greedy_search_recent_n1_s2.json"),
            os.path.join(PATHS.search, "greedy_search_joint_topk_n1.json"),
           ]

# %% define all files

dataf = {
    "rand-init-agg1": os.path.join(PATHS.search, "gpt2-rand-init_sce1_llen3_plen1_repeat_random_agg-1.json"),
    "rand-init-agg2": os.path.join(PATHS.search, "gpt2-rand-init_sce1_llen3_plen1_repeat_random_agg-2.json"),
    "rand-init-agg2-3": os.path.join(PATHS.search, "gpt2-rand-init_sce1_llen3_plen1_repeat_random_agg-2-3.json"),
    "rand-init-agg2-5": os.path.join(PATHS.search, "gpt2-rand-init_sce1_llen5_plen1_repeat_random_agg-2-5.json"),
    "rand-init-agg2-10": os.path.join(PATHS.search, "gpt2-rand-init_sce1_llen10_plen1_repeat_random_agg-2-10.json"),
    "ablate-all-agg1": os.path.join(PATHS.search, "gpt2-ablate-all_sce1_llen3_plen1_repeat_random_agg-1.json"),
    "ablate-all-agg2": os.path.join(PATHS.search, "gpt2-ablate-all_sce1_llen3_plen1_repeat_random_agg-2.json"),
    "ablate-all-agg2-3": os.path.join(PATHS.search, "gpt2-ablate-all_sce1_llen3_plen1_repeat_random_agg-2-3.json"),
    "ablate-all-agg2-5": os.path.join(PATHS.search, "gpt2-ablate-all_sce1_llen5_plen1_repeat_random_agg-2-5.json"),
    "ablate-all-agg2-10": os.path.join(PATHS.search, "gpt2-ablate-all_sce1_llen10_plen1_repeat_random_agg-2-10.json"),
    "unablated-agg1": os.path.join(PATHS.search, "gpt2_sce1_llen3_plen1_repeat_random.json"),
    "unablated-agg2": os.path.join(PATHS.search, "gpt2_sce1_llen3_plen1_repeat_random_agg-2.json"),
    "unablated-agg2-3": os.path.join(PATHS.search, "gpt2_sce1_llen3_plen1_repeat_random_agg-2-3.json"),
    "unablated-agg2-5": os.path.join(PATHS.search, "gpt2_sce1_llen5_plen1_repeat_random_agg-2-5.json"),
    "unablated-agg2-10": os.path.join(PATHS.search, "gpt2_sce1_llen10_plen1_repeat_random_agg-2-10.json"),
}

# %%

class GreedyOutput(object):

    def __init__(self) -> Any:
        
        self.rs = []
        self.rs_ci = []
        self.x1 = []
        self.x1_ci = []
        self.x2 = []
        self.x2_ci = []
        self.x = []
        self.xlab = []
        
    def __repr__(self):
        return str(f"GreedyOutput(N={len(self.rs)})\n"+"".join([f".{k}\n" for k in list(self.__dict__.keys())]))

    def load_json(self, filenames: List) -> Tuple:

        for f in filenames:

            with open(f, "r") as fh:

                d = json.load(fh)

                self.rs.append(d["rs"]["scores"])
                self.rs_ci.append(d["rs"]["ci"])
                self.x1.append(d["x1"]["scores"])
                self.x1_ci.append(d["x1"]["ci"])
                self.x2.append(d["x2"]["scores"])
                self.x2_ci.append(d["x2"]["ci"])
                self.x.append([len(e) for e in d["best_labels"]])
                self.xlab.append(d["best_labels"])

        self.rs = np.vstack(self.rs)
        self.rs_ci = np.stack(self.rs_ci)
        self.x1 = np.vstack(self.x1)
        self.x1_ci = np.stack(self.x1_ci)
        self.x2 = np.vstack(self.x2)
        self.x2_ci = np.stack(self.x2_ci)
        self.x = np.vstack(self.x)
        self.xlab = self.xlab

        return self

# %%

# load the search results
def load_json(filenames: List) -> Tuple:
    y, y_ci = [], []
    x1, x1_ci = [], []
    x2, x2_ci = [], []
    x, xlab = [], []

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
            xlab.append(d["best_labels"])

    yarr = np.vstack(y)
    y_ciarr = np.stack(y_ci)
    x1arr = np.vstack(x1)
    x1_ciarr = np.stack(x1_ci)
    x2arr = np.vstack(x2)
    x2_ciarr = np.stack(x2_ci)
    xarr = np.vstack(x)

    return (yarr, y_ciarr), (xarr,), (x1arr, x1_ciarr), (x2arr, x2_ciarr), xlab


# %%

def dict2mat(dct: Dict) -> np.ndarray:

    out = np.zeros(shape=(12, 12), dtype=bool)
    for layer in dct.keys():
        for head in dct[layer]:
            out[layer, head] = 1
    
    return out

# %%
def add_labels(ax, lst):

    for i, lab in enumerate(lst):
        dct = from_labels_to_dict([lab])
        layer_idx = list(dct.keys())[0]
        head_idx = list(dct.values())[0][0]
        ax.text(head_idx, layer_idx, str(i+1), ha="center", va="center", color="white", fontsize=9, fontweight="semibold")

    return ax


# %%
fn = os.path.join(PATHS.search, "gpt2_sce1_llen3_plen1_repeat_random.json")
with open(fn, "r") as fh:
    y_unab = json.load(fh)

# %%


def plot_search(greedy_output: GreedyOutput, ctrl_tup=None, rand_tup=None, abl_all_tup=None):

    xvals, yvals, yvals_ci = greedy_output.x.copy(), greedy_output.rs.copy(), greedy_output.rs_ci.copy()
    labels_list = greedy_output.xlab

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0::])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
    ax4 = fig.add_subplot(gs[1, 2], sharey=ax3)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    # if provided substract the unablated
    if ctrl_tup is not None:
        baseline, baseline_ci = ctrl_tup
        yvals = yvals - baseline
        yvals_ci[:, :, 0] = yvals_ci[:, :, 0] - baseline_ci[0]
        yvals_ci[:, :, 1] = yvals_ci[:, :, 1] - baseline_ci[1]

        ax1.text(x=14.5, y=5, s=f"*Unablated perf. = {baseline:.1f}%")

    ax1.plot(xvals[0, :], yvals[0, :], "o--", label="Matching", color=clrs.green)
    ax1.plot(xvals[1, :], yvals[1, :], "s--", label="Postmatch", color=clrs.blue)
    ax1.plot(xvals[2, :], yvals[2, :], "^--", label="Recent-tokens", color=clrs.orange)

    ax1.fill_between(xvals[0, :], y1=yvals_ci[0, :, 0], y2=yvals_ci[0, :, 1], color=clrs.green, alpha=0.3)
    ax1.fill_between(xvals[1, :], y1=yvals_ci[1, :, 0], y2=yvals_ci[1, :, 1], color=clrs.blue, alpha=0.3)
    ax1.fill_between(xvals[2, :], y1=yvals_ci[2, :, 0], y2=yvals_ci[2, :, 1], color=clrs.orange, alpha=0.3)

    xhlines = xvals[0, :]
    #if ctrl_tup is not None:
    #    ctrl, ci = ctrl_tup
    #    ax.hlines(ctrl, xhlines[0], xhlines[-1], linestyle="--", color="black", label="Unablated")
    #    ax.fill_between(xhlines, y1=ci[0], y2=ci[1], color="black", alpha=0.3)

    if rand_tup is not None:
        rand, ci = rand_tup
        if ctrl_tup is not None:
            rand = rand - baseline
            ci[0] = ci[0] - baseline_ci[0]
            ci[1] = ci[1] - baseline_ci[1]
        ax1.hlines(rand, xhlines[0], xhlines[-1], linestyle="-.", color="gray", label="Random init.")
        ax1.fill_between(xhlines, y1=ci[0], y2=ci[1], color="gray", alpha=0.3)

    if abl_all_tup is not None:
        abl_all, ci = abl_all_tup
        if ctrl_tup is not None:
            abl_all = abl_all - baseline
            ci[0] = ci[0] - baseline_ci[0]
            ci[1] = ci[1] - baseline_ci[1]
        ax1.hlines(abl_all, xhlines[0], xhlines[-1], linestyle=":", color="gray", label="All heads\nablated")
        ax1.fill_between(xhlines, y1=ci[0], y2=ci[1], color="gray", alpha=0.3)


    ax1.plot(xvals[0, np.argmax(yvals[0, :])], np.max(yvals[0, :]), marker="o", color="tab:red")
    ax1.plot(xvals[1, np.argmax(yvals[1, :])], np.max(yvals[1, :]), marker="s", color="tab:red")
    ax1.plot(xvals[2, np.argmax(yvals[2, :])], np.max(yvals[2, :]), marker="^", color="tab:red")

    find_at_th = lambda x, x_base, th: ((x-x_base)/(np.max(x-x_base)) > th)

    if ctrl_tup is not None:
        # plot 85% of max effect size
        th = 0.85
        cols = (clrs.green, clrs.blue, clrs.orange)
        lab = None
        for i in range(3):
            sel = find_at_th(yvals[i, :], np.min(yvals[i, :]), th)
            selv = yvals[i, sel][0]  # grab the first one that satisfies
            if i == 0:
                lab = f"> {int(th*100)}% of max\neffect"
            else:
                lab = None
            ax1.vlines(xvals[i, sel][0], 0, selv, linestyle="--", color="gray", label=lab)

    ax1.set_xlabel("Number of heads ablated (combination with largest effect)")
    ax1.set_ylabel("Median repeat surprisal (%)\n(change from unablated model*)")

    ax1.set_xticks(xvals[0, :])
    ax1.set_xticklabels(xvals[0, :])

    ax1.set_title("Greedy search per head type\n(define heads at `:`, evaluate at N1)")
    
    h, l = ax1.get_legend_handles_labels()

    if ctrl_tup is not None:
        legend1 = ax1.legend(h[3::], l[3::], bbox_to_anchor=(1.01, 0.5))
        ax1.add_artist(legend1)
        ax1.legend(h[0:3], l[0:3], title="Head type", bbox_to_anchor=(1.01, 1))
    else:
        ax1.legend(title="Head type", bbox_to_anchor=(1.01, 1))

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1.set_ylim(0, ax1.get_ylim()[1])
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

    # PLORT IMAGES
    y1_lab, y2_lab, y3_lab = labels_list  # order = matching, postmatch, recent-tokens
    lh_dict_1 = from_labels_to_dict(y1_lab[-1])
    im2 = ax2.imshow(dict2mat(lh_dict_1), cmap=plt.cm.Greens, origin="lower")
    add_labels(ax2, y1_lab[-1])
    ax2.set_title("Matching heads\n(importance order)")

    lh_dict_2 = from_labels_to_dict(y2_lab[-1])
    im3 = ax3.imshow(dict2mat(lh_dict_2), cmap=plt.cm.Blues, origin="lower")
    add_labels(ax3, y2_lab[-1])
    ax3.set_title("Post-match heads\n(importance order)")

    lh_dict_3 = from_labels_to_dict(y3_lab[-1])
    im4 = ax4.imshow(dict2mat(lh_dict_3), cmap=plt.cm.Oranges, origin="lower")
    add_labels(ax4, y3_lab[-1])
    ax4.set_title("Recent-tokens heads\n(importance order)")

    ax2.set_ylabel("Layer")
    fig.supxlabel("Head")

    ax2.set_yticks(np.arange(0, 12, 2))
    ax2.set_yticklabels(np.arange(1, 13, 2))
    for a in (ax2, ax3, ax4):
        a.set_xticks(np.arange(0, 12, 2))
        a.set_xticklabels(np.arange(1, 13, 2))

    return fig, (ax1, ax2, ax3, ax4)


# %%
def plot_search_x1x2(greedy_output: GreedyOutput, y3tup=None, y4tup=None):

    x1arr, x1_ciarr = greedy_output.x1.copy(), greedy_output.x1_ci.copy()
    x2arr, x2_ciarr = greedy_output.x2.copy(), greedy_output.x2_ci.copy()
    xarr = greedy_output.x.copy()

    if y3tup is not None:
        x1ctrl, x1ctrl_ci = y3tup[0], y3tup[1]
    if y4tup is not None:
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

    ax[1].fill_between(xarr[0, :], y1=x2_ciarr[0, :, 0], y2=x2_ciarr[0, :, 1], color=clrs.green, alpha=0.3)
    ax[1].fill_between(xarr[1, :], y1=x2_ciarr[1, :, 0], y2=x2_ciarr[1, :, 1], color=clrs.blue, alpha=0.3)
    ax[1].fill_between(xarr[2, :], y1=x2_ciarr[2, :, 0], y2=x2_ciarr[2, :, 1], color=clrs.orange, alpha=0.3)

    if y3tup is not None:
        unabl_x = np.arange(2.5, 21.5)
        ax[0].hlines(x1ctrl, 2.5, 20.5, linestyle="--", color="black", label="Unablated")
        ax[0].fill_between(unabl_x, y1=x1ctrl_ci[0], y2=x1ctrl_ci[1], color="black", alpha=0.3)

    if y4tup is not None:
        # plot control surprisal on x2
        ax[1].hlines(x2ctrl, 2.5, 20.5, linestyle="--", color="black", label="Unablated")
        ax[1].fill_between(unabl_x, y1=x2ctrl_ci[0], y2=x2ctrl_ci[1], color="black", alpha=0.3)

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

    return fig, ax


# %%

files_p1 = filenames[0:1] + filenames[3:4] + filenames[6:7]    # query = `:`
files_n1 = filenames[1:2] + filenames[4:5] + filenames[7:8]    # query = `N1`
files_n1_s2 = filenames[2:3] + filenames[5:6] + filenames[8:9] # query = `N1`, repeat surprisal over 2 tokens

# %%
# attending from `:`, evaluating repeat surprisal on N1
#y, x, x1, x2, xlab = load_json(filenames=files_p1)
ctrl_tup_x1 = (y_unab["x1"]["median"], y_unab["x1"]["ci95"])
ctrl_tup_x2 = (y_unab["x2"]["median"], y_unab["x2"]["ci95"])

# attending from N1, evaluating repeat surprisal on N2
#y_n1, x_n1, x1_n1, x2_n1, xlab_n1 = load_json(filenames=files_n1)
ctrl_tup_x1 = (y_unab["x1"]["median"], y_unab["x1"]["ci95"])
ctrl_tup_x2 = (y_unab["x2"]["median"], y_unab["x2"]["ci95"])

# attending from N1, evaluating repeat surprisal on N2 and N3
y_s2, x_s2, x1_s2, x2_s2, xlab_n2 = load_json(filenames=files_n1_s2)

savedir = os.path.join(PATHS.root, "fig", "ablation", "topk")

# %% on firs noun
greedy_n0 = GreedyOutput().load_json(files_p1)

with open(dataf["rand-init-agg1"], "r") as fh:
    y_rand_agg1 = json.load(fh)
with open(dataf["ablate-all-agg1"], "r") as fh:
    y_all_agg1 = json.load(fh)

fig, _ = plot_search(greedy_output=greedy_n0,
                  ctrl_tup=(y_unab["rs"]["median"], y_unab["rs"]["ci95"]),
                  rand_tup=(y_rand_agg1["rs"]["median"], y_rand_agg1["rs"]["ci95"]),
                  abl_all_tup=(y_all_agg1["rs"]["median"], y_all_agg1["rs"]["ci95"]),
                  )
                  
plt.tight_layout()
plt.show()
#fig.savefig(os.path.join(savedir, "greedy_search_p1_rs.png"), dpi=300)

# %%
fig = plot_search_x1x2(greedy_output=greedy_n0, y3tup=ctrl_tup_x1, y4tup=ctrl_tup_x2)
plt.show()
#fig.savefig(os.path.join(savedir, "greedy_search_p1_surp.png"), dpi=300)


#%%
greedy_n1 = GreedyOutput().load_json(files_n1)

with open(dataf["unablated-agg2"], "r") as fh:
    y_unab_agg2 = json.load(fh)
with open(dataf["rand-init-agg2"], "r") as fh:
    y_rand_agg2 = json.load(fh)
with open(dataf["ablate-all-agg2"], "r") as fh:
    y_all_agg2 = json.load(fh)

fig, axes = plot_search(greedy_output=greedy_n1,
                  ctrl_tup=(y_unab_agg2["rs"]["median"], y_unab_agg2["rs"]["ci95"]),
                  rand_tup=(y_rand_agg2["rs"]["median"], y_rand_agg2["rs"]["ci95"]),
                  abl_all_tup=(y_all_agg2["rs"]["median"], y_all_agg2["rs"]["ci95"]),
)

axes[0].set_title("Greedy search per head type\n(ablate at [N1], evaluate at [N2])")
plt.tight_layout()
plt.show()
#fig.savefig(os.path.join(savedir, "greedy_search_n1_rs.png"), dpi=300)

fig, ax = plot_search_x1x2(greedy_output=greedy_n1)
fig.suptitle("Greedy search -- raw surprisal scores\n(ablate at [N1], evaluate at [N2])")
plt.show()
#fig.savefig(os.path.join(savedir, "greedy_search_n1_surp.png"), dpi=300)


# %%

greedy_n1_s2 = GreedyOutput().load_json(files_n1_s2)

with open(dataf["unablated-agg2-3"], "r") as fh:
    y_unab_agg23 = json.load(fh)
with open(dataf["rand-init-agg2-3"], "r") as fh:
    y_rand_agg23 = json.load(fh)
with open(dataf["ablate-all-agg2-3"], "r") as fh:
    y_all_agg23 = json.load(fh)

fig, ax = plot_search(greedy_output=greedy_n1_s2,
                      ctrl_tup=(y_unab_agg23["rs"]["median"], y_unab_agg23["rs"]["ci95"]),
                      rand_tup=(y_rand_agg23["rs"]["median"], y_rand_agg23["rs"]["ci95"]),
                      abl_all_tup=(y_all_agg23["rs"]["median"], y_all_agg23["rs"]["ci95"]))

ax[0].set_title("Greedy search per head type\n(ablate at [N1], evaluate at: [N2, N3])")
plt.tight_layout()
plt.show()
#fig.savefig(os.path.join(savedir, "greedy_search_n1_s2_rs.png"), dpi=300)

fig, ax = plot_search_x1x2(greedy_output=greedy_n1_s2)
ax[0].set_title("Greedy search -- raw surprisal scores\n(ablate at [N1], evaluate at: [N2, N3])")
plt.show()
#fig.savefig(os.path.join(savedir, "greedy_search_n1_s2_surp.png"), dpi=300)

# %%

greedy_joint = GreedyOutput().load_json(filenames[-1::])

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(greedy_joint.x[0, :], greedy_joint.rs[0, :], "o--")
ax.fill_between(greedy_joint.x[0, :], y1=greedy_joint.rs_ci[0, :, 0], y2=greedy_joint.rs_ci[0, :, 1], alpha=0.3)

ax.set_xlabel("Number of heads ablated (combination with largest effect)")
ax.set_ylabel("Median repeat surprisal (%)")

ax.set_xticks(greedy_joint.x[0, :])
ax.set_xticklabels(greedy_joint.x[0, :])

ax.set_title("Joint search across matching/post-match/recent-tokens heads\n(define at `:`, evaluate at: [N1])")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

plt.tight_layout()
plt.show()

# %%

fs1 = glob(os.path.join(PATHS.search, "*all_heads_s1_iter-20.json"))
fs2 = glob(os.path.join(PATHS.search, "*all_heads_s2_iter-20.json"))
fs3 = glob(os.path.join(PATHS.search, "*all_heads_s3_iter-20.json"))

rs_s1, x_s1, x1_s1, x2_s1, xlab_s1 = load_json(fs1)
rs_s2, x_s2, x1_s2, x2_s2, xlab_s2 = load_json(fs2)
rs_s3, x_s3, x1_s3, x2_s3, xlab_s3 = load_json(fs3)

# %%
with open(fs[0], "r") as fh:
    d_s1 = json.load(fh)

with open(fs[1], "r") as fh:
    d_s2 = json.load(fh)

with open(fs[2], "r") as fh:
    d_s3 = json.load(fh)

# %%


# %%

fig = plt.figure(figsize=(7, 6))
gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[2, 1.3])
ax1 = fig.add_subplot(gs[0, 0::])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
ax4 = fig.add_subplot(gs[1, 2], sharey=ax3)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)

ax1.plot(np.arange(len(rs_s1[0][0, :])), rs_s1[0][0, :], "o--", label="First noun", color="tab:red")
ax1.plot(np.arange(len(rs_s2[0][0, :])), rs_s2[0][0, :], "o--", label="Second noun", color="tab:blue")
ax1.plot(np.arange(len(rs_s3[0][0, :])), rs_s3[0][0, :], "o--", label="Third noun", color="tab:green")

ax1.fill_between(np.arange(len(rs_s1[0][0, :])), y1=rs_s1[1][0, :, 0], y2=rs_s1[1][0, :, 1], color="tab:red", alpha=0.3)
ax1.fill_between(np.arange(len(rs_s2[0][0, :])), y1=rs_s2[1][0, :, 0], y2=rs_s2[1][0, :, 1], color="tab:blue", alpha=0.3)
ax1.fill_between(np.arange(len(rs_s3[0][0, :])), y1=rs_s3[1][0, :, 0], y2=rs_s3[1][0, :, 1], color="tab:green", alpha=0.3)

ax1.set_ylim([0, ax1.get_ylim()[1]])
ax1.set_xticks(np.arange(len(rs_s1[0][0, :])))
ax1.set_xticklabels(np.arange(1, len(rs_s1[0][0, :])+1))

ax1.set_xlabel("Number of heads ablated")
ax1.set_ylabel("Median repeat surprisal (%)")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

ax1.legend(title="Retrieval of")

ax1.set_title("Unconstrained greedy search across all heads\n(timeout after 12.5 hours of search)")

lh_dict_s1 = from_labels_to_dict(xlab_s1[0][19])
im5 = ax2.imshow(dict2mat(lh_dict_s1), cmap=plt.cm.Reds, origin="lower")

add_labels(ax2, xlab_s1[0][19])

lh_dict_s2 = from_labels_to_dict(xlab_s2[0][19])
im6 = ax3.imshow(dict2mat(lh_dict_s2), cmap=plt.cm.Blues, origin="lower")

add_labels(ax3, xlab_s2[0][19])

lh_dict_s3 = from_labels_to_dict(xlab_s3[0][19])
im7 = ax4.imshow(dict2mat(lh_dict_s3), cmap=plt.cm.Greens, origin="lower")

add_labels(ax4, xlab_s3[0][19])

ax2.set_ylabel("Layer")
ax2.set_yticks(np.arange(0, 12, 2))
ax2.set_yticklabels(np.arange(1, 13, 2))

for a in (ax2, ax3, ax4):
    a.set_xticks(np.arange(0, 12, 2))
    a.set_xticklabels(np.arange(1, 13, 2))

ax2.set_title("Effect @N1\n(importance order)")
ax3.set_title("Effect @N2\n(importance order)")
ax4.set_title("Effect @N3\n(importance order)")

fig.supxlabel("Head")

plt.tight_layout()
plt.show()

#fig.savefig(os.path.join(savedir, "greedy_search_unconstrained.png"), dpi=300)

# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True, sharey=True)

xvals = np.arange(len(x1_s1[0][0, :]))
line_x1_s1 = ax[0].plot(xvals, x1_s1[0][0, :], "o--", label="First noun", color="tab:red")
line_x1_s2 = ax[0].plot(xvals, x1_s2[0][0, :], "o--", label="Second noun", color="tab:blue")
line_x1_s3 = ax[0].plot(xvals, x1_s3[0][0, :], "o--", label="Third noun", color="tab:green")

fill_x1_s1 = ax[0].fill_between(xvals, y1=x1_s1[1][0, :, 0], y2=x1_s1[1][0, :, 1], color="tab:red", alpha=0.3)
fill_x1_s2 = ax[0].fill_between(xvals, y1=x1_s2[1][0, :, 0], y2=x1_s2[1][0, :, 1], color="tab:blue", alpha=0.3)
fill_x1_s3 = ax[0].fill_between(xvals, y1=x1_s3[1][0, :, 0], y2=x1_s3[1][0, :, 1], color="tab:green", alpha=0.3)

line_x2_s1 = ax[1].plot(xvals, x2_s1[0][0, :], "o--", label="First noun", color="tab:red")
line_x2_s2 = ax[1].plot(xvals, x2_s2[0][0, :], "o--", label="Second noun", color="tab:blue")
line_x2_s3 = ax[1].plot(xvals, x2_s3[0][0, :], "o--", label="Third noun", color="tab:green")

fill_x2_s1 = ax[1].fill_between(xvals, y1=x2_s1[1][0, :, 0], y2=x2_s1[1][0, :, 1], color="tab:red", alpha=0.3)
fill_x2_s2 = ax[1].fill_between(xvals, y1=x2_s2[1][0, :, 0], y2=x2_s2[1][0, :, 1], color="tab:blue", alpha=0.3)
fill_x2_s3 = ax[1].fill_between(xvals, y1=x2_s3[1][0, :, 0], y2=x2_s3[1][0, :, 1], color="tab:green", alpha=0.3)

for a in ax:
    a.set_xticks(xvals)
    a.set_xticklabels(np.arange(1, len(xvals)+1))

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.yaxis.set_minor_locator(AutoMinorLocator(2))
    a.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

ax[0].set_ylabel("Median surprisal (95% CI)")
fig.supxlabel("Search step (number of heads ablated)")

ax[0].legend(title="Surprisal computed on:")

ax[0].set_title("List 1")
ax[1].set_title("List 2")

fig.suptitle("Unconstrained greedy search across all heads  -- effect on raw surprisal scores\n(timeout after 12.5 hours of search)")

plt.tight_layout()
plt.show()

fig.savefig(os.path.join(savedir, "greedy_search_unconstrained_surp.png"), dpi=300)


# %%

fs5 = glob(os.path.join(PATHS.search, "*s2s5_iter-20*.json"))
fs10 = glob(os.path.join(PATHS.search, "*s2s10_iter-10*.json"))

rs_s5, x_s5, x1_s5, x2_s5, xlab_s5 = load_json(fs5)
rs_s10, x_s10, x1_s10, x2_s10, xlab_s10 = load_json(fs10)

with open(fs5[0], "r") as fh:
    d_s5 = json.load(fh)

with open(fs10[0], "r") as fh:
    d_s10 = json.load(fh)

with open(dataf["rand-init-agg2-5"], "r") as fh:
    y_rand_agg25 = json.load(fh)
with open(dataf["ablate-all-agg2-5"], "r") as fh:
    y_all_agg25 = json.load(fh)
with open(dataf["rand-init-agg2-10"], "r") as fh:
    y_rand_agg10 = json.load(fh)
with open(dataf["ablate-all-agg2-10"], "r") as fh:
    y_all_agg10 = json.load(fh)


# %%

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0::])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)

xvals = np.arange(len(rs_s5[0][0, :]))
ax1.plot(xvals, rs_s5[0][0, :], "o--", label="[N2-N5]", color="tab:red")
ax1.plot(np.arange(len(rs_s10[0][0, :])), rs_s10[0][0, :], "o--", label="[N2-N10]", color="tab:blue")

ax1.fill_between(xvals, y1=rs_s5[1][0, :, 0], y2=rs_s5[1][0, :, 1], color="tab:red", alpha=0.3)
ax1.fill_between(np.arange(len(rs_s10[0][0, :])), y1=rs_s10[1][0, :, 0], y2=rs_s10[1][0, :, 1], color="tab:blue", alpha=0.3)

# plot random init
ax1.hlines(y_rand_agg25["rs"]["median"], 0, xvals[-1], linestyle="--", color="tab:red", label="Random init")
ax1.hlines(y_rand_agg10["rs"]["median"], 0, xvals[-1], linestyle="--", color="tab:blue", label="Random init")

ci = y_rand_agg25["rs"]["ci95"]
ax1.fill_between(xvals, y1=ci[0], y2=ci[1], color="tab:red", alpha=0.3)
ci = y_rand_agg10["rs"]["ci95"]
ax1.fill_between(xvals, y1=ci[0], y2=ci[1], color="tab:red", alpha=0.3)

# plot ablate all
ax1.hlines(y_all_agg25["rs"]["median"], 0, xvals[-1], linestyle="-.", color="tab:red", label="Ablate all")
ax1.hlines(y_all_agg10["rs"]["median"], 0, xvals[-1], linestyle="-.", color="tab:blue", label="Ablate all", zorder=2)

ci = y_all_agg25["rs"]["ci95"]
ax1.fill_between(xvals, y1=ci[0], y2=ci[1], color="tab:red", alpha=0.3)
ci = y_all_agg10["rs"]["ci95"]
ax1.fill_between(xvals, y1=ci[0], y2=ci[1], color="tab:red", alpha=0.3)

ax1.set_ylim([0, ax1.get_ylim()[1]])
ax1.set_xticks(xvals)
ax1.set_xticklabels(np.arange(1, len(xvals)+1))

ax1.set_xlabel("Number of heads ablated")
ax1.set_ylabel("Median repeat surprisal (%)")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

ax1.legend(title="Retrieval of")

ax1.set_title("Unconstrained greedy search across all heads\n(timeout after 14 hours of search)")

lh_dict_s5 = from_labels_to_dict(xlab_s5[0][19])
im5 = ax2.imshow(dict2mat(lh_dict_s5), cmap=plt.cm.Reds, origin="lower")

add_labels(ax2, xlab_s5[0][19])

lh_dict_s10 = from_labels_to_dict(xlab_s10[0][9])
im6 = ax3.imshow(dict2mat(lh_dict_s10), cmap=plt.cm.Blues, origin="lower")

add_labels(ax3, xlab_s10[0][9])

for a in (ax2, ax3):
    a.set_xticks(np.arange(0, 12, 2))
    a.set_xticklabels(np.arange(1, 13, 2))
ax2.set_yticks(np.arange(0, 12, 2))
ax2.set_yticklabels(np.arange(1, 13, 2))

fig.supxlabel("Head")
ax2.set_ylabel("Layer")

ax1.legend(title="Surprisal computed on:", bbox_to_anchor=[1, 1])

ax2.set_title("Found heads at each\nsearch step\n(N2-N5)")
ax3.set_title("Found heads at each\nsearch step\n(N2-N10)")

plt.tight_layout()
plt.show()

#fig.savefig(os.path.join(savedir, "greedy_search_unconstrained_s5-s10.png"), dpi=300)
# %%


fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True, sharey=True)

xvals = np.arange(len(x1_s5[0][0, :]))
xvals2 = np.arange(len(x1_s10[0][0, :]))
line_x1_s1 = ax[0].plot(xvals, x1_s5[0][0, :], "o--", label="[N2-N5]", color="tab:red")
line_x1_s2 = ax[0].plot(xvals2, x1_s10[0][0, :], "o--", label="[N2-N10]", color="tab:blue")

fill_x1_s1 = ax[0].fill_between(xvals, y1=x1_s5[1][0, :, 0], y2=x1_s5[1][0, :, 1], color="tab:red", alpha=0.3)
fill_x1_s2 = ax[0].fill_between(xvals2, y1=x1_s10[1][0, :, 0], y2=x1_s10[1][0, :, 1], color="tab:blue", alpha=0.3)

line_x2_s1 = ax[1].plot(xvals, x2_s5[0][0, :], "o--", color="tab:red")
line_x2_s2 = ax[1].plot(xvals2, x2_s10[0][0, :], "o--", color="tab:blue")

fill_x2_s1 = ax[1].fill_between(xvals, y1=x2_s5[1][0, :, 0], y2=x2_s5[1][0, :, 1], color="tab:red", alpha=0.3)
fill_x2_s2 = ax[1].fill_between(xvals2, y1=x2_s10[1][0, :, 0], y2=x2_s10[1][0, :, 1], color="tab:blue", alpha=0.3)

for a in ax:
    a.set_xticks(xvals)
    a.set_xticklabels(np.arange(1, len(xvals)+1))

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.yaxis.set_minor_locator(AutoMinorLocator(2))
    a.grid(visible=True, which="both", linewidth=0.5, linestyle="--")

ax[0].set_ylabel("Median surprisal (95% CI)")
fig.supxlabel("Search step (number of heads ablated)")

ax[0].legend(title="Surprisal computed on:")

ax[0].set_title("List 1")
ax[1].set_title("List 2")

fig.suptitle("Unconstrained greedy search across all heads  -- effect on raw surprisal scores\n(timeout after 14 hours of search)")

plt.tight_layout()
plt.show()

fig.savefig(os.path.join(savedir, "greedy_search_unconstrained_s5-s10_surp.png"), dpi=300)

# %%

find_at = lambda x, x_base, th: ((x-x_base)/(np.max(x-x_base)) > th)

def get_val_n(yin, xin, baseline, th):
    sel = find_at(yin-baseline, np.min(yin-baseline), th)
    y_sel = yin[sel][0]
    x_sel = xin[sel][0]

    return f"{y_sel:.0f}% (N={x_sel})"

thresh = 0.85
b1 = y_unab["rs"]["median"]
match_n1 = get_val_n(y[0][0, :], x[0][0, :], b1, thresh)
postmatch_n1 = get_val_n(y[0][1, :], x[0][1, :], b1, thresh)
recent_n1 = get_val_n(y[0][2, :], x[0][2, :], b1, thresh)

b2 = y_unab_agg2["rs"]["median"]
match_n2 = get_val_n(y_n1[0][0, :], x_n1[0][0, :], b2, thresh)
postmatch_n2 = get_val_n(y_n1[0][1, :], x_n1[0][1, :], b2, thresh)
recent_n2 = get_val_n(y_n1[0][2, :], x_n1[0][2, :], b2, thresh)

unconstrained_n1 = get_val_n(rs_s1[0][0, :], x_s1[0][0, :], b1, thresh)
unconstrained_n2 = get_val_n(rs_s2[0][0, :], x_s2[0][0, :], b2, thresh) 


#%%
row_names = ["Matching", "Post-match", "Recent", "Unconstrained", "Rand. init", "Ablate all"]
col_names = ["First noun", "Second noun"]

df = pd.DataFrame(index=row_names, columns=col_names, dtype=str)

df.loc["Matching", "First noun"] = match_n1
df.loc["Post-match", "First noun"] = postmatch_n1
df.loc["Recent", "First noun"] = recent_n1

df.loc["Matching", "Second noun"] = match_n2
df.loc["Post-match", "Second noun"] = postmatch_n2
df.loc["Recent", "Second noun"] = recent_n2

df.loc["Unconstrained", "First noun"] = unconstrained_n1
df.loc["Rand. init", "First noun"] = f"{y_rand_agg1['rs']['median']:.1f}%"
df.loc["Unconstrained", "Second noun"] = unconstrained_n2
df.loc["Rand. init", "Second noun"] = f"{y_rand_agg2['rs']['median']:.1f}%"
df.loc["Ablate all", "First noun"] = f"{y_all_agg1['rs']['median']:.1f}%"
df.loc["Ablate all", "Second noun"] = f"{y_all_agg2['rs']['median']:.1f}%"


# %%
df = df.T
df.index.name = "Eval time-step"
df.columns.name = "Search type"
print(df.to_latex(escape=True, bold_rows=True,
                    caption="Ablation results for constrained and unconstrained greedy search "
                    "evaluated at the first and second noun in the list. "
                      "Reported are results for the combination of ablated heads that "
                      "achieved at lest 85\% of the largest found effect."))

# %%

# at n1
match_labs_n1 = xlab[0][17]
postmatch_labs_n1 = xlab[1][17]
recent_labs_n1 = xlab[2][17]

match_labs_n2 = xlab_n2[0][19]
postmatch_labs_n2 = xlab_n2[1][19]
recent_labs_n2 = xlab_n2[2][19]

uncon_n1 = xlab_s1[0][19]
uncon_n2 = xlab_s2[0][19]

# %%

n1_common = set(match_labs_n1) & set(postmatch_labs_n1) & set(recent_labs_n1) & set(uncon_n1)
n2_common = set(match_labs_n2) & set(postmatch_labs_n2) & set(recent_labs_n2) & set(uncon_n2)

# %%
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
              

col_names = ["Matching", "Post-match", "Recent", "Matching & Postmatch & Recent"]
row_names = ["Unconstrained"]

df2 = pd.DataFrame(index=row_names, columns=col_names, dtype=str)

df2.loc["Unconstrained", "Matching"] = ", ".join(set(match_labs_n1) & set(uncon_n1))
df2.loc["Unconstrained", "Post-match"] = ", ".join(set(postmatch_labs_n1) & set(uncon_n1))
df2.loc["Unconstrained", "Recent"] = ", ".join(set(recent_labs_n1) & set(uncon_n1))
df2.loc["Unconstrained", "Matching & Postmatch & Recent"] = ", ".join(n1_common) if n1_common else "None"

print(df2.to_latex(bold_rows=True, caption="Commmon attention heads found by contrained and unconstrained search."
                   "Evaluated at first nonun in the list"))

# %%

col_names = ["Matching", "Post-match", "Recent", "Matching & Postmatch & Recent"]
row_names = ["Unconstrained"]

df3 = pd.DataFrame(index=row_names, columns=col_names, dtype=str)

df3.loc["Unconstrained", "Matching"] = ", ".join(set(match_labs_n2) & set(uncon_n2))
df3.loc["Unconstrained", "Post-match"] = ", ".join(set(postmatch_labs_n2) & set(uncon_n2))
df3.loc["Unconstrained", "Recent"] = ", ".join(set(recent_labs_n2) & set(uncon_n2))
df3.loc["Unconstrained", "Matching & Postmatch & Recent"] = ", ".join(n2_common) if n2_common else "None"

print(df3.to_latex(bold_rows=True, multirow=True, caption="Common attention heads found by constrained and unconstrained search. "
                   "Evaluated at second nonun in the list"))

# %%
