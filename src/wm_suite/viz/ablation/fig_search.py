
# %%
import os
import json
import configparser
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, List, Tuple, Union

from paths import PATHS
from wm_suite.wm_ablation import from_labels_to_dict
from wm_suite.viz.ablation.inputs import get_filenames

# %%

class GreedyOutput(object):


    def __init__(self) -> None:
        
        pass

    def load_json(datadir, fnames):

        pass



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

def make_plot(data: Dict, member_dict: Dict):

    fig = plt.figure(figsize=(7, 6))

    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 2, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 1])

    x1 = np.arange(0, len(data["rs"]["scores"]))
    y1 = np.array(data["rs"]["scores"])
    yE = np.array(data["rs"]["ci"])  # shape = (2, len(x1))

    ax1.plot(x1, y1, "--o", label="First noun")
    ax1.fill_between(x1, y1=yE[:, 0], y2=yE[:, 1], alpha=0.2)

    ax1.set_xticks(np.arange(len(member_dict)))
    tmp = data["best_labels"][-1]
    xlabels = np.array(["" for _ in range(len(member_dict))], dtype=object)
    for i, s in enumerate(tmp):
        xlabels[i] = f"{i+1} " + reformat_str(s)

    ax1.set_xticklabels(xlabels, rotation=45, ha="right")

    ax1.set_ylim(0, 100 if ax1.get_ylim()[1] < 100 else ax1.get_ylim()[1])
    ax1.set_xlabel("Found head")
    ax1.set_ylabel("Median repeat surprisal (%)")

    cols = {0:'tab:green', 1:'tab:blue', 2:'tab:orange'}
    for i, l in enumerate(ax1.xaxis.get_ticklabels()):
        if l.get_text() != "":
            lab = data["best_labels"][-1][i]
            l.set_color(cols[member_dict[lab]])


    # image plot
    imd = make_matrix(data["best_labels"][-1])
    ax2.imshow(imd, cmap=plt.cm.Greys, origin="lower")

    coords = [label2coord(l) for l in data["best_labels"][-1]]
    for i, c in enumerate(coords):

        if i < 9:
            xind = c[1] - 0.25
        else:
            xind = c[1] - 0.5
        ax2.text(x=xind, y=c[0]-0.25, s=f"{i+1}", color="white", fontweight="semibold")
        patch_color = cols[member_dict[data["best_labels"][-1][i]]]
        ax2.add_patch(Rectangle((c[1]-0.5, c[0]-0.5), 1, 1, fill=False, edgecolor=patch_color, lw=2))


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

    plt.tight_layout()
    
    return fig, (ax1, ax2)


# %%

files = get_filenames(os.path.basename(__file__))

data1 = load_json(os.path.join(PATHS.search, files["topk_n1"]))
data2 = load_json(os.path.join(PATHS.search, files["topk_n2"]))

data1_ = load_json(os.path.join(PATHS.search, files["topk_n1_list"]))
data2_ = load_json(os.path.join(PATHS.search, files["topk_n2_list"]))

membership1 = get_membership_dict((data1_["lh_list"], data1_["members"]))
membership2 = get_membership_dict((data2_["lh_list"], data2_["members"]))


# %%

fig, axes = make_plot(data1, membership1)
plt.show()

# %%

fig2, axes2 = make_plot(data2, membership2)
plt.show()
# %%
