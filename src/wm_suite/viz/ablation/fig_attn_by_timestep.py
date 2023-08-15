#%%

import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib import colormaps as cmaps
from matplotlib.ticker import AutoMinorLocator

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
def mowing_window_attention(attn_all, target_ids):

    out = []
    for i, tgt_id in enumerate(target_ids):

        out.append(np.mean(attn_all[:, tgt_id, :, i], axis=0))

    return np.array(out)


#%%
datadir = os.path.join(PATHS.data, "ablation")
d0 = load_data(datadir, "p1")[-1]
d1 = load_data(datadir, "n1")[-1]["data"]
d2 = load_data(datadir, "n2")[-1]["data"]
d3 = load_data(datadir, "n3")[-1]["data"]
d4 = load_data(datadir, "n4")[-1]["data"]
d5 = load_data(datadir, "n5")[-1]["data"]
d6 = load_data(datadir, "n6")[-1]["data"]
d7 = load_data(datadir, "n7")[-1]["data"]
d8 = load_data(datadir, "n8")[-1]["data"]
d9 = load_data(datadir, "n9")[-1]["data"]
d10 = load_data(datadir, "n10")[-1]["data"]


# %%
# define the toi
x = d0["data"]  # use the data sampled at ':' to define the heads
lh_dict_match, _, match_vals = find_topk_attn(x, topk=10, tokens_of_interest=[13], seed=12345)
lh_dict_post, _, post_vals = find_topk_attn(x, topk=10, tokens_of_interest=[14], seed=12345)
lh_dict_recent, _, recent_vals = find_topk_attn(x, topk=10, tokens_of_interest=[58], seed=12345)
# %%

# stack attention weights for all nouns
all_x = np.stack([d0['data'], d1, d2, d3, d4, d5, d6, d7, d8, d9, d10], axis=-1)

#%%

ids_match = get_ids_from_dict(lh_dict_match)
ids_post = get_ids_from_dict(lh_dict_post)
ids_recent = get_ids_from_dict(lh_dict_recent)

x_match, x_match_labels = select_topk_heads(all_x, ids_match)
x_post, x_post_labels = select_topk_heads(all_x, ids_post)
x_recent, x_recent_labels = select_topk_heads(all_x, ids_recent)

# %%

match = 13
onset_query = 59
nouns_2 = np.arange(onset_query+1, onset_query+20, 2)
match_nouns = np.arange(match+1, match+20, 2)
postmatch_commas = np.arange(match+2, match+21, 2)
postmatch_nouns = np.arange(match+3, match+22, 2)
recent_tokens = nouns_2-1

# %%

x_match_sel = mowing_window_attention(attn_all=x_match, target_ids=match_nouns)
x_post_sel = mowing_window_attention(attn_all=x_post, target_ids=postmatch_nouns)
x_recent_sel = mowing_window_attention(attn_all=x_recent, target_ids=recent_tokens)

# %%

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(np.mean(x_match_sel, axis=0), "--o", color=clrs.green, label="Matching nouns in list 1")
ax.plot(np.mean(x_post_sel, axis=0), "--o", color=clrs.blue, label="Post-match tokens in list 1")
ax.plot(np.mean(x_recent_sel, axis=0), "--o", color=clrs.orange, label="Previous token")

ax.set_xlabel("Noun position in second list")
ax.set_xticks(np.arange(0, 10, 1))
ax.set_xticklabels(np.arange(1, 11, 1))

ax.set_ylabel("Mean attention weight")
ax.set_title("Attention from list 2 nouns broken by position in list")

ax.legend(title="Attention to:")

plt.tight_layout()
plt.show()

# %%

def fetch_data(x_match, sequence, frame):

    if (type(sequence) == str) & (sequence == "all"):
        selseq = np.arange(x_match.shape[0])
    else:
        selseq = np.array([sequence])

    datnan = np.full(x_match.shape, np.nan)
    if len(selseq) > 1:
        for s in selseq:
            hasvalue = np.where(x_match[s, :, 0, frame] > 0.00)[0]
            datnan[s, hasvalue, :, frame] = x_match[s, hasvalue, :, frame]
    else:
        hasvalue = np.where(x_match[selseq.item(), :, 0, frame] > 0.00)[0]
        datnan[selseq, hasvalue, :, frame] = x_match[selseq, hasvalue, :, frame]

    # if averaging over sequences, create generic token streing
    if len(selseq) > 1:
        toks = np.array(['<|endoftext|>', 'Before', 'the', 'meeting', ',', 'Mary', 'wrote', 'down', 'the', 'following', 'list', 'of', 'words', ':', 
         'N1', ',', 'N2', ',', 'N3', ',', 'N4', ',', 'N5', ',', 'N6', ',', 'N7', ',', 'N8', ',', 'N9', ',', 'N10', '.',
         'After', 'the', 'meeting', ',', 'she,', 'took', 'a', 'break', 'and', 'had', 'a', 'cup', 'of', 'coffee', '.', 'When', 'she', 'got', 'back', ',',
         'she', 'read', 'the', 'list', 'again', ':', 'N1', ',', 'N2', ',', 'N3', ',', 'N4', ',', 'N5', ',', 'N6', ',', 'N7', ',', 'N8', ',', 'N9', 
         ',', 'N10', ".", "<|endoftext|>"])
        toks = toks[hasvalue]
    else:
        print(selseq)
        print(hasvalue)
        toks = [s.strip('Ġ') for s in d0['tokens'][selseq.item()][hasvalue]]

    if len(selseq) == 1:
        return datnan[selseq, :, :, frame], toks, hasvalue
    elif len(selseq) > 1:
        return np.mean(datnan[selseq, :, :, frame], axis=0), toks.tolist(), hasvalue


# %%

def find_onsets(tokens, which):

    onset = len(tokens)-1
    ons = np.where(np.array(tokens) == ":")[0]

    if which == "matching":
        delta = ons[1]-ons[0]
    elif which == "postmatch":
        delta = (ons[1]-ons[0])-1
    elif which == "recent":
        delta = 1

    return onset, onset-delta

# %%

plotwhich = "postmatch"

if plotwhich == "matching":
    x_dat = x_match
    title = "Attention timecourse of top-10 matching heads"
    savname = "matching_timecourse.gif"
    current_cmap = cmaps["Greens"]
    lc="tab:green"
    ylabs = x_match_labels

elif plotwhich == "postmatch":
    x_dat = x_post
    title = "Attention timecourse of top-10 post-match heads"
    savname = "postmatch_timecourse.mp4"
    current_cmap = cmaps["Blues"]
    lc="tab:blue"
    ylabs = x_post_labels

elif plotwhich == "recent":
    x_dat = x_recent
    title = "Attention timecourse of top-10 recent-tokens heads"
    savname = "recent_timecourse.gif"
    current_cmap = cmaps["Oranges"]
    lc="tab:orange"
    ylabs = x_recent_labels

current_cmap.set_bad(color='lightgrey')


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 4.5), sharex=True, 
                       gridspec_kw={'height_ratios': [1, 2]})

nframe = 0
datnan, seltok, hasvals = fetch_data(x_dat, sequence="all", frame=nframe)

patch_ons = find_onsets(seltok, plotwhich)

im = ax[1].imshow(datnan.T, aspect="auto", vmin=0, vmax=1, cmap=current_cmap)

liney = np.median(datnan[0:patch_ons[1]+1, :], axis=1)
line1 = ax[0].plot(liney, "--o", color=lc, markersize=6)[0]

ax[1].set_ylabel("Head\n(Layer/head index)")
ax[1].set_xlabel("Token")
ax[1].set_xticks(hasvals)
ax[1].set_xticklabels(seltok, rotation=90)
ax[1].set_yticks(np.arange(0, 10))
ax[1].set_yticklabels(ylabs)

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].grid(axis='both', which="both", linewidth=0.5)
ax[0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0].set_ylabel("Med. att.\nacross heads")

rect1 = patches.Rectangle((patch_ons[0]-0.5, -0.5), 1, 10, linewidth=1, ec='tab:purple', facecolor='none', zorder=2)
rect2 = patches.Rectangle((patch_ons[1]-0.5, -0.5), 1, 10, linewidth=1, ec='tab:purple', facecolor='none', zorder=2)

ax[1].add_patch(rect1)
ax[1].add_patch(rect2)

def find_minmax(data, timestep):
    maxatt = np.max(data[timestep, :])
    minatt = np.min(data[timestep, :])
    medatt = np.median(data[timestep, :])
    return minatt, maxatt, medatt

setstr = lambda x: f"Min = {x[0]:.2f}\nMed = {x[2]:.2f}\nMax = {x[1]:.2f}"

txtoffset = 8
mina, maxa, meda = find_minmax(datnan, patch_ons[1])
text = ax[1].text(x=patch_ons[1]-txtoffset, y=3, s=setstr((mina, maxa, meda)))

cax = ax[1].inset_axes([1.02, 0.0, 0.02, 1.0])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.set_ylabel("Mean attention weight")

fig.suptitle(title)

def update(frame):

    data, tokens, hasval = fetch_data(x_dat, sequence="all", frame=frame)
    im.set_data(data.T)

    rect_onsets = find_onsets(tokens, plotwhich)

    linedat = np.median(data[0:rect_onsets[1]+1, :], axis=1)

    line1.set_ydata(linedat)
    line1.set_xdata(np.arange(0, len(linedat)))
    rect1.set_x(rect_onsets[0]-0.5)
    rect2.set_x(rect_onsets[1]-0.5)

    ax[1].add_patch(rect1)
    ax[1].add_patch(rect2)

    mina, maxa, meda = find_minmax(data, rect_onsets[1])
    text.set_text(setstr((mina, maxa, meda)))
    text.set_x(rect_onsets[1]-txtoffset)

    ax[1].set_xticks(hasval)
    ax[1].set_xticklabels(tokens, rotation=90)

    return im,

plt.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=11, interval=1500, blit=True)
plt.show()

# %%

savedir = os.path.join(PATHS.root, "fig", "attn")
logging.info(f"Saving to {os.path.join(savedir, f'{savname}')}")
ani.save(os.path.join(savedir, f"{savname}"))


# %%
if __name__ == "__main__":
    pass