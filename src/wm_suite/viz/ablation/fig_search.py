# %%
import os
import json
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, List, Tuple

from wm_suite.paths import get_paths
from wm_suite.viz.ablation.inputs import get_filenames
from wm_suite.viz.utils import save_png_pdf
from wm_suite.viz.func import set_manuscript_style


PATHS = get_paths()
# %%

def load_json(fname:str) -> Dict:

    with open(fname, "r") as f:
        data = json.load(f)

    return data


# %%
def label2coord(label:str) -> Tuple[int, int]:

    s1, s2 = label.split(".")

    return (int(s1.strip("L")), int(s2.strip("H")))


# %%
def make_matrix(labels: List) -> np.ndarray:

    x = np.zeros(shape=(12, 12), dtype=bool)

    for label in labels:
        
        xy = label2coord(label)
        x[xy[0], xy[1]] = True

    return x

# %%
def get_membership_dict(datain: Tuple) -> Dict:

    lh_list, members = datain

    return {s: idx for s, idx in zip(lh_list, members)}


# %%
def reformat_str(s:str) -> str:

    s1, s2 = s.split(".")
    return f"L{int(s1.strip('L'))+1}.H{int(s2.strip('H'))+1}"


# %%
def plot_heads(axis, labels, member_cols, member_dict):

    # image plot
    imd = make_matrix(labels)
    axis.imshow(imd, cmap=plt.cm.Greys, origin="lower")

    coords = [label2coord(l) for l in labels]
    for i, c in enumerate(coords):

        if i < 9:
            xind = c[1] - 0.25
        else:
            xind = c[1] - 0.5
        axis.text(x=xind, y=c[0]-0.25, s=f"{i+1}", color="white", fontweight="semibold")
        patch_color = member_cols[member_dict[labels[i]]]
        axis.add_patch(Rectangle((c[1]-0.5, c[0]-0.5), 1, 1, fill=False, edgecolor=patch_color, lw=2))

    return axis

# %%

def plot_search_trajectory(data: Dict, member_dict: Dict, member_cols: Dict = None):

    fig = plt.figure(figsize=(7, 6))

    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 2, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 1])

    x1 = np.arange(0, len(data["rs"]["scores"]))
    y1 = np.array(data["rs"]["scores"])
    yE = np.array(data["rs"]["ci"])  # shape = (2, len(x1))

    ax1.plot(x1, y1, "--o", label="Attention-based\nablation")
    ax1.fill_between(x1, y1=yE[:, 0], y2=yE[:, 1], alpha=0.2)

    ax1.set_xticks(np.arange(len(member_dict)))
    tmp = data["best_labels"][-1]
    xlabels = np.array(["" for _ in range(len(member_dict))], dtype=object)
    for i, s in enumerate(tmp):
        xlabels[i] = f"{i+1}  " + reformat_str(s)

    ax1.set_xticklabels(xlabels, rotation=90)

    ax1.set_ylim(0, 100 if ax1.get_ylim()[1] < 100 else ax1.get_ylim()[1])
    ax1.set_xlabel("Ablated head")
    ax1.set_ylabel("Median repeat surprisal\n(%)")

    if member_cols is None:
        member_cols = {0:'tab:green', 1:'tab:blue', 2:'tab:orange'}
    for i, l in enumerate(ax1.xaxis.get_ticklabels()):
        if l.get_text() != "":
            lab = data["best_labels"][-1][i]
            l.set_color(member_cols[member_dict[lab]])


    ax2 = plot_heads(ax2, data["best_labels"][-1], member_cols, member_dict)

    ax2.set_xticks(np.arange(0, 12, 1))
    ax2.set_xticklabels([i if i % 2 != 0 else ""for i in range(1, 13)])
    ax2.set_yticks(np.arange(0, 12, 1))
    ax2.set_yticklabels([i if i % 2 != 0 else "" for i in range(1, 13)])
    ax2.set_ylabel("Layer")
    ax2.set_xlabel("Head")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(visible=True, axis="both", linestyle="--", alpha=0.5)
    ax2.grid(visible=True, axis="both", linestyle="--", alpha=0.5)
    
    return fig, (ax1, ax2)


# %%
def get_random_search_results(files):

    rnd, rnd_lab = [], []
    for f in files:
        with open(os.path.join(PATHS.search, f), "r") as fh:
            dct = json.load(fh)
            rnd.append(dct["rs"]["scores"])
            rnd_lab.append(dct["best_labels"])

    n_possible_searches = 43
    rnd_mat = np.full(shape=(len(rnd), n_possible_searches), fill_value=np.nan, dtype=float)
    for i, r in enumerate(rnd):
        rnd_mat[i, :len(r)] = r

    return rnd_mat, rnd_lab


# %%
def plot_attention_based_with_random(data, data_rnd, membership):

    fig, axes = plot_search_trajectory(data, membership)

    axes[0].plot(data_rnd.T, '-', color="gray", alpha=0.55, lw=0.7)

    def round_to_nearest(x, base=10):
        return int(base * round(float(x)/base))

    ymax = np.nanmax(data_rnd)+10
    ytickmax = round_to_nearest(ymax, 10)
    print(ytickmax)

    axes[0].set_ylim(0, ymax)
    axes[0].set_yticks(np.arange(0, ytickmax, 10))
    axes[0].set_yticklabels([i if i%20 == 0 else "" for i in np.arange(0, ytickmax, 10)])
    axes[0].set_title("Greedy search trajectory, attention-informed search\n(effects on N2+N3)")
    axes[1].set_title("Searched heads")

    return fig, axes


# %%
def search_across_all_heads(data, rand_init, ablate_all, membership: Dict):

    memb_dict = {**{k: 3 for k in data["best_labels"][-1]}, **membership}

    fig6, axes6 = plot_search_trajectory(data, memb_dict, {0: "tab:green", 1: "tab:blue", 2: "tab:orange", 3: "tab:grey"})
    axes6[0].set_title("Greedy search results across all heads (effects on N2+N3)")
    axes6[1].set_title("Searched heads")

    axes6[0].hlines(rand_init["rs"]["median"], 0, 20, linestyle="--", color="tab:purple", label="Rand. init.")
    axes6[0].hlines(ablate_all["rs"]["median"], 0, 20, linestyle="-.", color="tab:red", label="All heads ablated")

    ci = rand_init["rs"]["ci95"]
    axes6[0].fill_between(np.arange(20), y1=ci[0], y2=ci[1], color="tab:purple", alpha=0.3)
    ci = ablate_all["rs"]["ci95"]
    axes6[0].fill_between(np.arange(20), y1=ci[0], y2=ci[1], color="tab:red", alpha=0.3)
    axes6[0].legend(title="Model type")

    return fig6, axes6

# %%
def make_plot(datadir, which:str):

    files = get_filenames(os.path.basename(__file__))


    if which == "attention_based_with_random":

        data5 = load_json(os.path.join(datadir, files["topk_n2_n3_orig"]))
        data4_ = load_json(os.path.join(datadir, files["topk_n1_list_orig"]))   # aggregated across 3-tokens etc.
        membership3 = get_membership_dict((data4_["lh_list"], data4_["members"]))

        data_rnd, _ = get_random_search_results(files["random_neg"])

        fig, axes = plot_attention_based_with_random(data5, data_rnd, membership3)
        plt.tight_layout()

    elif which == "across_all_heads":
        
        rand_n1_n2 = load_json(os.path.join(PATHS.search, files["n2_n3_rand-init"]))
        all_n1_n2 = load_json(os.path.join(PATHS.search, files["n2_n3_ablate-all"]))

        data6 = load_json(os.path.join(PATHS.search, files["all_n2_n3"]))
        data4_ = load_json(os.path.join(datadir, files["topk_n1_list_orig"]))   # aggregated across 3-tokens etc.
        membership3 = get_membership_dict((data4_["lh_list"], data4_["members"]))

        fig, axes = search_across_all_heads(data=data6, rand_init=rand_n1_n2, ablate_all=all_n1_n2, membership=membership3)
        plt.tight_layout()

    return fig, axes


# %%

def main(input_args=None):
    
    import argparse

    parser = argparse.ArgumentParser(description="Plot search results")
    parser.add_argument("-d", "--datadir", type=str, help="path to data directory")
    parser.add_argument("-w", "--which", type=str)
    parser.add_argument("-s", "--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    elif isinstance(input_args, list):
        args = parser.parse_args(input_args)

    set_manuscript_style()
    if args.which == "attention_based_n2_n3":

        fig, _ = make_plot(args.datadir, args.which)
        plt.show()

        if args.savedir:
            save_png_pdf(fig, os.path.join(args.savedir, "fig_search_attention_based"))

    if args.which == "across_all_heads":

        make_plot(datadir=args.datadir, which="across_all_heads")
        plt.show()

        if args.savedir:
            save_png_pdf(fig, os.path.join(args.savedir, "search_across_all_heads"))

    pass


# %%

if __name__ == "__main__":

    main()
