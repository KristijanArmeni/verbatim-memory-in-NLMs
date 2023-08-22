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

datadict = {}
infix_strings1 = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10"]
infix_strings2 = ["r1", "r2", "r3", "r4", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

p1_tokens = load_data(datadir, "p1")[-1]["tokens"]  # load the dict with all keys for one dataset
datadict["p1"] = load_data(datadir, "p1")[-1]["data"]  # load the `data` field

for infix in infix_strings1+infix_strings2:
    
    # load the `data` field
    datadict[infix] = load_data(datadir, infix)[-1]["data"]


# %%
# define the toi
x = datadict["p1"]  # use the data sampled at ':' to define the heads
lh_dict_match, _, match_vals = find_topk_attn(x, topk=10, tokens_of_interest=[13], seed=12345)
lh_dict_post, _, post_vals = find_topk_attn(x, topk=10, tokens_of_interest=[14], seed=12345)
lh_dict_recent, _, recent_vals = find_topk_attn(x, topk=10, tokens_of_interest=[58], seed=12345)
# %%

# stack attention weights for all nouns
stackorder = ["r4", "r3", "r2", "r1", "p1", 
              "n1", "c1", "n2", "c2", "n3", "c3", 
              "n4", "c4", "n5", "c5", "n6", "c6", 
              "n7", "c7", "n8", "c8", "n9", "c9", "n10"]

tostack = [datadict[infix] for infix in stackorder]
all_x = np.stack(tostack, axis=-1)

#%%

ids_match = get_ids_from_dict(lh_dict_match)
ids_post = get_ids_from_dict(lh_dict_post)
ids_recent = get_ids_from_dict(lh_dict_recent)

x_match, x_match_labels = select_topk_heads(all_x, ids_match)
x_post, x_post_labels = select_topk_heads(all_x, ids_post)
x_recent, x_recent_labels = select_topk_heads(all_x, ids_recent)


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
        toks = [s.strip('Ġ') for s in p1_tokens[selseq.item()][hasvalue]]

    if len(selseq) == 1:
        return datnan[selseq, :, :, frame], toks, hasvalue
    elif len(selseq) > 1:
        return np.mean(datnan[selseq, :, :, frame], axis=0), toks.tolist(), hasvalue


# %%

def find_onsets(tokens, which):

    onset = len(tokens)-1
    ons = np.where(np.array(tokens) == ":")[0]

    if which == "matching":
        if len(ons) == 1:
            delta = None
            delta2 = None
        else:
            delta = ons[1]-ons[0]
            delta2 = ons[1]-ons[0]
    elif which == "postmatch":
        if len(ons) == 1:
            delta = None
            delta2 = None
        else:
            delta = (ons[1]-ons[0])-1
            delta2 = ons[1]-ons[0]
    elif which == "recent":
        delta = 1
        delta2 = None
        if len(ons) > 1:
            delta2 = ons[1] - ons[0]

    if delta is not None:
        target_onset = onset-delta  # timestep of the target token
    
    if delta2 is not None:
        match_onset = onset-delta2

    if (delta is None) & (which in ["matching", "postmatch"]):
        outtuple = (onset,)
    elif (which == "recent") & (delta2 is None):
        outtuple = (onset, target_onset)
    elif delta2 is not None:
        outtuple = (onset, target_onset, match_onset)

    return outtuple

# %%

plotwhich = "matching"

if plotwhich == "matching":
    x_dat = x_match
    title = "Attention timecourse of top-10 matching heads"
    savname = "matching_timecourse"
    current_cmap = cmaps["Greens"]
    lc="tab:green"
    ylabs = x_match_labels

elif plotwhich == "postmatch":
    x_dat = x_post
    title = "Attention timecourse of top-10 post-match heads"
    savname = "postmatch_timecourse"
    current_cmap = cmaps["Blues"]
    lc="tab:blue"
    ylabs = x_post_labels

elif plotwhich == "recent":
    x_dat = x_recent
    title = "Attention timecourse of top-10 recent-tokens heads"
    savname = "recent_timecourse"
    current_cmap = cmaps["Oranges"]
    lc="tab:orange"
    ylabs = x_recent_labels

current_cmap.set_bad(color='lightgrey')


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 4.5), sharex=True, 
                       gridspec_kw={'height_ratios': [1, 2]})

nframe = 0
datnan, seltok, hasvals = fetch_data(x_dat, sequence="all", frame=nframe)

# get reference attention from frame 4 (':')
#ref_frame = 4
#ref_dat, ref_tok, ref_hasvals = fetch_data(x_dat, sequence="all", frame=4)
#ref_vals = ref_dat[13, :]

# check if there are two patches to be drawn or just one
def set_timesteps(onset_ids):

    marker1_ons = onset_ids[0]
    marker2_ons = None
    target_timestep = None
    
    if len(onset_ids) > 1:
        target_timestep = onset_ids[1]
    
    if len(onset_ids) == 3:
        marker2_ons = onset_ids[2]

    return marker1_ons, target_timestep, marker2_ons

def find_minmax(data, timestep):
    maxatt = np.max(data[timestep, :])
    minatt = np.min(data[timestep, :])
    medatt = np.median(data[timestep, :])
    return minatt, maxatt, medatt

setstr = lambda x: f"Min = {x[0]:.2f}\nMed = {x[2]:.2f}\nMax = {x[1]:.2f}"

patch_ons = find_onsets(seltok, plotwhich)
marker1_ons, patch_timestep, marker2_ons = set_timesteps(patch_ons)

im = ax[1].imshow(datnan.T, aspect="auto", vmin=0, vmax=1, cmap=current_cmap)

liney = np.median(datnan[0:marker1_ons+1, :], axis=1)
line1 = ax[0].plot(liney, "--o", color=lc, markersize=5)[0]

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

#rect1 = patches.Rectangle((marker1_ons-0.5, -0.5), 1, 10, linewidth=1, ec='tab:purple', facecolor='none', zorder=2)
#ax[1].add_patch(rect1)
marker1 = ax[1].vlines(marker1_ons, 8.5, 9.5, color='tab:red', linewidth=2)

# draw it at marker1_ons initially, but hide and activate once patch_ons exists
rect = patches.Rectangle((marker1_ons-0.5, -0.5), 1, 10, linewidth=1, ec='tab:purple', facecolor='none', zorder=2)  
ax[1].add_patch(rect)
rect.set_visible(False)

if marker2_ons is None:
    marker2 = ax[1].vlines(marker1_ons, 8.5, 9.5, color='tab:red', linewidth=2)
    marker2.set_visible(False)
elif marker2_ons is not None:
    marker2 = ax[1].vlines(marker2_ons, 8.5, 9.5, color='tab:red', linewidth=2)
    marker2.set_visible(True)

txtoffset = 8
text = ax[1].text(x=marker1_ons-txtoffset, y=3, s="")

cax = ax[1].inset_axes([1.02, 0.0, 0.02, 1.0])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.set_ylabel("Mean attention weight")

fig.suptitle(title)

def update(frame):

    data, tokens, hasval = fetch_data(x_dat, sequence="all", frame=frame)

    im.set_data(data.T)

    rect_onsets = find_onsets(tokens, plotwhich)

    marker1_ons, patch_timestep, marker2_ons = set_timesteps(rect_onsets)

    linedat = np.median(data[0:marker1_ons+1, :], axis=1)

    line1.set_ydata(linedat)
    line1.set_xdata(np.arange(0, len(linedat)))

    segs = marker1.get_segments()[0]
    segs[:, 0] = marker1_ons
    marker1.set_segments([segs])

    if patch_timestep is not None:
        rect.set_x(patch_timestep-0.5)
        rect.set_visible(True)
        mina, maxa, meda = find_minmax(data, patch_timestep)
        text.set_text(setstr((mina, maxa, meda)))
        text.set_x(patch_timestep-txtoffset)

    if marker2_ons is not None:
        segs2 = marker2.get_segments()[0]
        segs2[:, 0] = marker2_ons
        marker2.set_segments([segs2])
        marker2.set_visible(True)

    ax[1].set_xticks(hasval)
    ax[1].set_xticklabels(tokens, rotation=90)

    return im,

plt.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=24, interval=1500, blit=True)
plt.show()

# %%

savedir = os.path.join(PATHS.root, "fig", "attn")

savname_html = os.path.join(savedir, f"{savname}.html")
logging.info(f"Saving to {savname_html}")
with open(os.path.join(savedir, f"{savname_html}"), "w") as f:
    print(ani.to_jshtml(), file=f)

savname_gif = os.path.join(savedir, f"{savname}.gif")
logging.info(f"Saving to {savname_gif}")
ani.save(os.path.join(savedir, f"{savname_gif}"))


#%%

def prepare_timesteps(x_in, keystr, frames):

    v = {"attn":[], "labs": [], "query": [], "query_id": [], "target_id": [], "target": []}

    for f in frames:
        x, t, h = fetch_data(x_in, sequence="all", frame=f)
        endt, targett, _ = find_onsets(t, keystr)
        #print(f"from: {t[endt]} ({endt}) | to: {t[targett]}) ({targett})")

        v["attn"].append(x[targett, :])
        v["labs"].append(f"{t[targett]}\n({targett})")
        v["query"].append(t[endt])
        v["query_id"].append(endt)
        v["target"].append(t[targett])
        v["target_id"].append(targett)

    
    return v

odd_frames = [4] + list(range(5, 24, 2))
even_frames = [4] + list(range(6, 24, 2))

# noun frames
dm1 = prepare_timesteps(x_match, "matching", odd_frames)
dp1 = prepare_timesteps(x_post, "postmatch", even_frames)
dr1 = prepare_timesteps(x_recent, "recent", even_frames)

dm2 = prepare_timesteps(x_match, "matching", even_frames)
dp2 = prepare_timesteps(x_post, "postmatch", odd_frames)
dr2 = prepare_timesteps(x_recent, "recent", odd_frames)


#%%

fig, ax = plt.subplots(2, 3, figsize=(15, 6), sharey=True)

ax[0, 0].plot(np.arange(len(dm1["labs"])), dm1["attn"], "--o", color=clrs.green)
ax[0, 1].plot(np.arange(len(dp1["labs"])), dp1["attn"], "--^", color=clrs.blue)
ax[0, 2].plot(np.arange(len(dr1["labs"])), dr1["attn"], "--s", color=clrs.orange)

ax[0, 0].set_xticks(np.arange(len(dm1["labs"])))
ax[0, 0].set_xticklabels(dm1["labs"])
ax[0, 1].set_xticks(np.arange(len(dp1["labs"])))
ax[0, 1].set_xticklabels(dp1["labs"])
ax[0, 2].set_xticks(np.arange(len(dr1["labs"])))
ax[0, 2].set_xticklabels(dr1["labs"])

ax[1, 0].plot(np.arange(len(dm2["labs"]))[1::], dm2["attn"][1::], "--o", color=clrs.green)
ax[1, 1].plot(np.arange(len(dp2["labs"]))[1::], dp2["attn"][1::], "--^", color=clrs.blue)
ax[1, 2].plot(np.arange(len(dr2["labs"]))[1::], dr2["attn"][1::], "--s", color=clrs.orange)

ax[1, 0].set_xticks(np.arange(len(dm2["labs"]))[1::])
ax[1, 0].set_xticklabels(dm2["labs"][1::])
ax[1, 1].set_xticks(np.arange(len(dp2["labs"]))[1::])
ax[1, 1].set_xticklabels(dp2["labs"][1::])
ax[1, 2].set_xticks(np.arange(len(dr2["labs"]))[1::])
ax[1, 2].set_xticklabels(dr2["labs"][1::])


ax[0, 0].set_title("Top-10 matching heads\n(target token occurs once)")
ax[0, 1].set_title("Top-10 post-match heads\n(target token occurs once)")
ax[0, 2].set_title("Top-10 recent-tokens heads\n(target token occurs once)")

ax[1, 0].set_title("Top-10 matching heads\n(target token occurs at multiple positions)")
ax[1, 1].set_title("Top-10 post-match heads\n(target token occurs at multiple positions)")
ax[1, 2].set_title("Top-10 recent-tokens heads\n(target token occurs at multiple positions)")

for a in ax[0, :]:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(visible=True, which="both", linewidth=0.5)

for a in ax[1, :]:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(visible=True, which="both", linewidth=0.5)

fig.supxlabel("Target token in list\n(Timestep index in input sequence)")
fig.supylabel("Mean attention across input sequences")

plt.tight_layout()


# %%
if __name__ == "__main__":
    pass