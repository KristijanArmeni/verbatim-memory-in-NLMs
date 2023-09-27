#%% 

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from paths import PATHS
from src.wm_suite.viz.ablation.fig_attn import get_data, find_topk_attn
from src.wm_suite.viz.func import set_manuscript_style
from src.wm_suite.viz.utils import save_png_pdf, clrs

import logging
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO, format="%(message)s")


# %%
def load_data(datadir, token_string:str) -> Tuple[np.ndarray, Dict]:

    fn = os.path.join(datadir, f"attention_weights_gpt2_colon-colon-{token_string}-ctxlen1_llen10.npz")
    data, full_dict = get_data(fn)

    return data, full_dict


#%%
def get_ids_from_dict(lh_dict: Dict) -> List[Tuple[int, int]]:
    """
    Convert the dictionary of layer-heads to a list of tuples of layer-heads.

    Parameters
    ----------
    lh_dict : Dict
        Dictionary of layer-heads.

    Returns
    -------
    ids : List[Tuple[int, int]]
        List of tuples of layer-head indices (e.g. `(0, 2)` for layer 0 and head 2).
    """
    ids = [(l, h) for l in lh_dict.keys() if lh_dict[l] for h in lh_dict[l]]

    return ids


#%%
def select_topk_heads(attn: np.ndarray, coords: List[Tuple]):

    """
    Loops through layer-head indices in coords tuples and returns the attention scores for those layer-heads.

    Parameters
    ----------
    attn : np.ndarray (sequences, timesteps, heads, layers, noun_timesteps)
        Attention scores for all layer-heads.
    
    """

    xout = np.zeros(shape=attn.shape[0:2] + (len(coords),) + attn.shape[-1:])
    labs = []
    for i, tup in enumerate(coords):
        
        layer_id, head_id = tup[0], tup[1]
        xout[:, :, i, :] = attn[:, :, head_id, layer_id, :]
        labs.append(f"L{layer_id+1}.H{head_id+1}")
    
    return xout, labs # shape = (sequences, tokens, heads, nouns)


# %%
datadir = os.path.join(PATHS.data, "ablation")

datadict = {}
infix_strings1 = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10"]
#infix_strings2 = ["r1", "r2", "r3", "r4", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

p1_tokens = load_data(datadir, "p1")[-1]["tokens"]  # load the dict with all keys for one dataset
datadict["p1"] = load_data(datadir, "p1")[-1]["data"]  # load the `data` field

for infix in infix_strings1:
    
    # load the `data` field
    datadict[infix] = load_data(datadir, infix)[-1]["data"]

# %%

tostack = [datadict[infix] for infix in datadict.keys()]
all_x = np.stack(tostack, axis=-1)  # shape (sequences, tokens, heads, layers, queries)

# %%

# define the toi
x = datadict["p1"]  # use the data sampled at ':' to define the heads
lh_dict_match, _, match_vals = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=[13], seed=12345)
lh_dict_post, _, post_vals = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=[14, 16, 18], seed=12345)
lh_dict_recent, _, recent_vals = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=[58, 57, 56], seed=12345)

# %%

ids_match = get_ids_from_dict(lh_dict_match)
ids_post = get_ids_from_dict(lh_dict_post)
ids_recent = get_ids_from_dict(lh_dict_recent)

unselected = [(i, j) for i in range(12) for j in range(12) if (i, j) not in ids_match+ids_post+ids_recent]

x_match, x_match_labels = select_topk_heads(all_x, ids_match)
x_post, x_post_labels = select_topk_heads(all_x, ids_post)
x_recent, x_recent_labels = select_topk_heads(all_x, ids_recent)
x_unselected, x_unselected_labels = select_topk_heads(all_x, unselected)

# %%
# 13 == `:`, 14 == `N1`, 15 == `,`, 16 == `N2` etc. 
n1_match_id, n1_query_id = 14, 60
n2_match_id, n2_query_id = 16, 62
n3_match_id, n3_query_id = 18, 64
n4_match_id, n4_query_id = 20, 66

# a pedestian check
assert p1_tokens[0][n1_query_id-1] == ":"
assert p1_tokens[0][n1_query_id] == "Ġpatience"
assert p1_tokens[0][n1_match_id] == "Ġpatience"

ag_func = np.median

match_y1 = ag_func(x_match, axis=0)[n1_match_id, :, 1]   # attend to `n1` from `n1`
match_y2 = ag_func(x_match, axis=0)[n2_match_id, :, 2]   # attend to `n2` from n2
match_y3 = ag_func(x_match, axis=0)[n3_match_id, :, 3]   # attend to `n3` from n3
match_y4 = ag_func(x_match, axis=0)[n4_match_id, :, 4]   # attend to `n4` from n4

post_y1 = ag_func(x_post, axis=0)[n1_match_id+1, :, 1]   # attend to `n2` from `n1`
post_y2 = ag_func(x_post, axis=0)[n2_match_id+1, :, 2]   # attend to `n3` from n2
post_y3 = ag_func(x_post, axis=0)[n3_match_id+1, :, 3]   # attend to `n4` from n3
post_y4 = ag_func(x_post, axis=0)[n4_match_id+1, :, 4]   # attend to `n5` from n4

recent_y1 = ag_func(x_recent, axis=0)[n1_query_id-1, :, 1]   # attend to `:` from `n1`
recent_y2 = ag_func(x_recent, axis=0)[n2_query_id-1, :, 2]   # attend to `,` from 'n2'
recent_y3 = ag_func(x_recent, axis=0)[n3_query_id-1, :, 3]   # attend to `,` from n3
recent_y4 = ag_func(x_recent, axis=0)[n4_query_id-1, :, 4]   # attend to `,` from n4

unselected_y1 = ag_func(x_unselected, axis=0)[n1_match_id, :, 1]   # attend to `n1` from `n1`
unselected_y2 = ag_func(x_unselected, axis=0)[n2_match_id, :, 2]   # attend to `n2` from n2
unselected_y3 = ag_func(x_unselected, axis=0)[n3_match_id, :, 3]   # attend to `n3` from n3
unselected_y4 = ag_func(x_unselected, axis=0)[n4_match_id, :, 4]   # attend to `n4` from n4

unselected_y1_B = ag_func(x_unselected, axis=0)[n1_match_id+1, :, 1]   # attend to `n1` from `n1`
unselected_y2_B = ag_func(x_unselected, axis=0)[n2_match_id+1, :, 2]   # attend to `n2` from n2
unselected_y3_B = ag_func(x_unselected, axis=0)[n3_match_id+1, :, 3]   # attend to `n3` from n3
unselected_y4_B = ag_func(x_unselected, axis=0)[n4_match_id+1, :, 4]   # attend to `n4` from n4



# %%

df_match = pd.DataFrame({"Position = 1": match_y2, "Position = 2": match_y3, "Position = 3": match_y4, "Attention to": "Matching token (list 1)"})
df_post = pd.DataFrame({"Position = 1": post_y2, "Position = 2": post_y3, "Position = 3": post_y4, "Attention to": "Postmatch token (list 1)"})
df_recent = pd.DataFrame({"Position = 1": recent_y2, "Position = 2": recent_y3, "Position = 3": recent_y4, "Attention to": "Recent token (list 2)"})

df = pd.concat([df_match, df_post, df_recent], axis=0)

# %%

set_manuscript_style()

fig, ax = plt.subplots(1, 4, figsize=(14, 5), sharey=True, sharex=True)

labelfs = 22

ax[0].scatter(match_y1, match_y2,
              s=120, alpha=0.8, edgecolor="None", marker="o", linewidth=0.5, color=clrs.green)
ax[0].scatter(post_y1, post_y2,
              s=120, alpha=0.8, edgecolor="None", marker="s", linewidth=0.5, color=clrs.blue)
ax[0].scatter(recent_y1, recent_y2,
              s=120, alpha=0.8, edgecolor="None", marker="D", linewidth=0.5, color=clrs.orange)

ax[0].set_xlabel("Position = 1", fontsize=labelfs, color="gray")
ax[0].set_ylabel("Position = 2", fontsize=labelfs, color="gray")

ax[1].scatter(match_y2, match_y3,
              s=120, alpha=0.8, edgecolor="None", marker="o", linewidth=0.5, color=clrs.green)
ax[1].scatter(post_y2, post_y3, s=120, alpha=0.8, edgecolor="None", marker="s", linewidth=0.5, color=clrs.blue)
ax[1].scatter(recent_y2, recent_y3, s=120, alpha=0.8, edgecolor="None", marker="D", linewidth=0.5, color=clrs.orange)

ax[1].set_ylabel("Position = 3", fontsize=labelfs, color="gray")
ax[1].set_xlabel("Position = 2", fontsize=labelfs, color="gray")

ax[2].scatter(match_y2, match_y4, label="Matching token (list 1)",
              s=120, alpha=0.8, edgecolor="None", marker="o", linewidth=0.5, color=clrs.green)
ax[2].scatter(post_y2, post_y4, label="Postmatch token (list 1)",
              s=120, alpha=0.8, edgecolor="None", marker="s", linewidth=0.5, color=clrs.blue)
ax[2].scatter(recent_y2, recent_y4, label="Recent token (list 2)",
              s=120, alpha=0.8, edgecolor="None", marker="D", linewidth=0.5, color=clrs.orange)

ax[2].set_xlabel("Position = 2", fontsize=labelfs, color="gray")
ax[2].set_ylabel("Position = 4", fontsize=labelfs, color="gray")

ax[3].scatter(match_y3, match_y4,
              s=120, alpha=0.8, edgecolor="None", marker="o", linewidth=0.5, color=clrs.green)
ax[3].scatter(post_y3, post_y4,
              s=120, alpha=0.8, edgecolor="None", marker="s", linewidth=0.5, color=clrs.blue)
ax[3].scatter(recent_y3, recent_y4,
              s=120, alpha=0.8, edgecolor="None", marker="D", linewidth=0.5, color=clrs.orange)

ax[3].set_xlabel("Position = 3", fontsize=labelfs, color="gray")
ax[3].set_ylabel("Position = 4", fontsize=labelfs, color="gray")

for a in ax:
    a.set_aspect("equal")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_yticklabels([0, "", "", "", "", 0.5, "", "", "", "", 1.0], fontsize=labelfs)
    a.set_xticklabels([0, "", "", "", "", 0.5, "", "", "", "", 1.0], fontsize=labelfs)
    a.grid(visible=True, axis="both", linestyle="--", color="lightgray")

fig.legend(title="Attention to", loc="upper right", 
              bbox_to_anchor=(1.1, 0.5), frameon=True,
              fontsize=14, title_fontsize=14)


fig.supxlabel("Attention", ha="center", fontsize=labelfs)
fig.supylabel("Attention", ha="center", fontsize=labelfs)

fig.suptitle("How consistent is attention from different positions in list 2?", 
             fontsize=labelfs+1, fontweight="semibold")

plt.tight_layout()
plt.show()

# %%

save_png_pdf(fig, savename=os.path.join(PATHS.root, "fig", "attn", "fig_attn_scatter"))

# %%

set_manuscript_style()

gr = sns.pairplot(df, hue="Attention to", markers=["o", "s", "D"], corner=False, diag_kind="kde", aspect=1, height=3.2,
                  palette={"Matching token (list 1)": "tab:green", 
                           "Postmatch token (list 1)": "tab:blue", 
                           "Recent token (list 2)": "tab:orange"},
                  plot_kws={"s": 120, "alpha": 0.8, "edgecolor": "None", "linewidth": 0.5},
                  diag_kws={"cut": 0})

fig = plt.gcf()

l, h = fig.axes[7].get_legend_handles_labels()
for ax in fig.axes[1:3] + fig.axes[5:6]:
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for collection in ax.collections:
        collection.remove()

for a in fig.axes[0:1] + fig.axes[3:5] + fig.axes[6::]:

    a.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_yticklabels([0, "", "", "", "", 0.5, "", "", "", "", 1.0], fontsize=28)
    a.set_xticklabels([0, "", "", "", "", 0.5, "", "", "", "", 1.0], fontsize=28)
    a.grid(visible=True, axis="both", linestyle="--", color="lightgray")

leg = fig.axes[2].legend(l, h, title="Attention to", fontsize=20, title_fontsize=20, markerscale=3)
fig.axes[2].add_artist(leg)
gr._legend.remove()

#gr.map_diag(sns.histplot(kde=True, cut=0))

for a in fig.axes:
    a.set_ylabel(a.get_ylabel(), fontsize=28, color="gray")
    a.set_xlabel(a.get_xlabel(), fontsize=28, color="gray")
fig.supxlabel("Attention", ha="center", fontsize=28)
fig.supylabel("Attention", ha="center", fontsize=28)

plt.tight_layout()
plt.show()

# %%
save_png_pdf(fig, savename=os.path.join(PATHS.root, "fig", "attn", "fig_attn_scatter"))

# %%
