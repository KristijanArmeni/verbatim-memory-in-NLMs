#%% 

import os
import numpy as np
from matplotlib import pyplot as plt

from wm_suite.paths import get_paths
from wm_suite.utils import logger
from wm_suite.viz.ablation.inputs import get_filenames
from wm_suite.viz.ablation.fig_attn import get_data, find_topk_attn
from wm_suite.viz.func import set_manuscript_style
from wm_suite.viz.utils import save_png_pdf, clrs
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger("wm_suite.utils")

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
def find_attn_heads_at_colon(x):
    
    # define the toi
    lh_dict_match, _, match_vals = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=[13], seed=12345)
    lh_dict_post, _, post_vals = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=[14, 16, 18], seed=12345)
    lh_dict_recent, _, recent_vals = find_topk_attn(x, topk=20, attn_threshold=0.2, tokens_of_interest=[58, 57, 56], seed=12345)

    return lh_dict_match, lh_dict_post, lh_dict_recent


# %%
def extract_attn_vals_per_position(x_match: np.ndarray, x_post: np.ndarray, x_recent: np.ndarray, p1_tokens: List[str]) -> Tuple:

    # 13 == `:`, 14 == `N1`, 15 == `,`, 16 == `N2` etc. 
    n1_match_id, n1_query_id = 14, 60
    n2_match_id, n2_query_id = 16, 62
    n3_match_id, n3_query_id = 18, 64
    n4_match_id, n4_query_id = 20, 66

    # a pedestian check that our indices find the expected tokens
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

    return (match_y1, match_y2, match_y3, match_y4), (post_y1, post_y2, post_y3, post_y4), (recent_y1, recent_y2, recent_y3, recent_y4)


# %%
def make_scatterplots(match_tuple: Tuple, post_tuple:Tuple, recent_tuple:Tuple):

    match_y1, match_y2, match_y3, match_y4 = match_tuple
    post_y1, post_y2, post_y3, post_y4 = post_tuple
    recent_y1, recent_y2, recent_y3, recent_y4 = recent_tuple

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

    return fig, ax


# %%

def make_plot(datadir):

    files = get_filenames(os.path.basename(__file__))

    # ===== DATA ===== #
    datadict = {}
    #infix_strings1 = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10"]

    p1_tokens = get_data(os.path.join(datadir, files["p1"]))[-1]["tokens"]  # load the dict with all keys for one dataset
    datadict["p1"] = get_data(os.path.join(datadir, files["p1"]))[-1]["data"]  # load the `data` field

    tostack = [get_data(os.path.join(datadir, files[key]))[-1]["data"] for key in files.keys()]

    all_x = np.stack(tostack, axis=-1)  # shape (sequences, tokens, heads, layers, queries)

    # find attention heads at the colon position
    lh_dict_match, lh_dict_post, lh_dict_recent = find_attn_heads_at_colon(x=datadict["p1"])

    ids_match = get_ids_from_dict(lh_dict_match)
    ids_post = get_ids_from_dict(lh_dict_post)
    ids_recent = get_ids_from_dict(lh_dict_recent)

    # select the same attention heads across all positions
    x_match, x_match_labels = select_topk_heads(all_x, ids_match)
    x_post, x_post_labels = select_topk_heads(all_x, ids_post)
    x_recent, x_recent_labels = select_topk_heads(all_x, ids_recent)

    # now pull out attention scores from n1, n2, n3 and n4
    match_tuple, post_tuple, recent_tuple = extract_attn_vals_per_position(x_match, x_post, x_recent, p1_tokens)

    # ===== MAKE THE FIGURE ===== #
    fig, ax = make_scatterplots(match_tuple, post_tuple, recent_tuple)
    plt.tight_layout()

    return fig, ax


# %%

def main(input_args: List[str] = None):

    import argparse

    parser = argparse.ArgumentParser(description="Plot attention weights across token positions")
    parser.add_argument("-d", "--datadir", type=str, help="path to data directory")
    parser.add_argument("-s", "--savedir", type=str, help="path where to save the figure (.pdf and .png)")

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    set_manuscript_style()

    fig, _ = make_plot(args.datadir)
    plt.show()

    if args.savedir:
        fn = os.path.join(args.savedir, "fig_attn_scatter")
        save_png_pdf(fig, savename=args.savedir)

    pass

# %%

if __name__ == "__main__":

    main()
