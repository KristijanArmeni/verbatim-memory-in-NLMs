# %%
import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from paths import PATHS
from src.wm_suite.wm_ablation import find_topk_attn, get_pairs, from_dict_to_labels
from src.wm_suite.viz.func import filter_and_aggregate
from src.wm_suite.viz.utils import save_png_pdf
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

#%%

def pairs2list(attn, which):

    if which == "postmatching":
        toi = [14, 16, 18]
    elif which == "matching":
        toi = [13]
    elif which == "recent":
        toi = [44, 43, 42]

    lh_dict, _, _ = find_topk_attn(attn, topk=20, tokens_of_interest=toi, seed=12345)

    pairs_list = ["-".join([s.replace(".", "") for s in from_dict_to_labels(e)]) for e in get_pairs(lh_dict)]

    return pairs_list


def reformat_pairstr(pairstr:str) -> str:

    str1 = pairstr.split("-")[0]
    str2 = pairstr.split("-")[1]

    d1 = int(str1.split("H")[0].strip("L")), int(str1.split("H")[-1])
    d2 = int(str2.split("H")[0].strip("L")), int(str2.split("H")[-1])

    return f"L{d1[0]:02}H{d1[1]:02}-L{d2[0]:02}H{d2[1]:02}"

#%%  

def save_pair_labels_to_json(savedir):

    attn_data = os.path.join(PATHS.data, "ablation\\attention_weights_gpt2_colon-colon-p1.npz")
    attn = dict(np.load(attn_data))["data"]

    pairs_postmatching = pairs2list(attn, which="postmatching")
    fnjson = os.path.join(savedir, "postmatching-top20-pairs.json")
    with open(fnjson, "w") as fh:
        json.dump(pairs_postmatching, fh)

    pairs_matching = pairs2list(attn, which="matching")
    fnjson = os.path.join(savedir, "matching-top20-pairs.json")
    with open(fnjson, "w") as fh:
        json.dump(pairs_matching, fh)

    pairs_recent = pairs2list(attn, which="recent")
    fnjson = os.path.join(savedir, "recent-top20-pairs.json")
    with open(fnjson, "w") as fh:
        json.dump(pairs_recent, fh)

    return 0

# %%

#savedir = os.path.join(PATHS.data, 'ablation', 'zero_attn', 'topk', 'pairs')
#save_pair_labels_to_json(savedir)

#%% load_files()
def load_files(datadir, pairlabels, which: str):

    f = []
    for p in tqdm(pairlabels, desc="file"):

        fn = f"ablation_gpt2_{which}-{p}_sce1_1_3_random_repeat.csv"
        tmp = pd.read_csv(os.path.join(datadir, fn), sep="\t")
        tmp['model'] = "gpt2"
        tmp['second_list'] = "repeat"
        tmp['list'] = "random"
        tmp['prompt_len'] = 8
        tmp = tmp.rename(columns={'trialID': 'marker', "stimID": "stimid"})
        tmp['context'] = "intact"
        tmp['model_id'] = reformat_pairstr(p)
        
        f.append(tmp)

    return f

#%%
#datadir = "C:\\users\\karmeni1\\project\\lm-mem\\data\\ablation\\zero_attn\\topk\\pairs"

#with open("C:\\users\\karmeni1\project\\lm-mem\\data\\attn\\postmatching-top20-pairs.json") as fh:
#    pairs_postmatching = json.load(fh)
#dfs = load_files(datadir, pairlabels=pairs_postmatching, which="postmatching")


# %% COMPUTE REPEAT SURPRISALS
def get_repeat_surprisals(dataframes):
    
    # grab the first time step (technically, mean computation is applied, but no effective here)
    variables = [{"list_len": [3]},
                {"prompt_len": [8]},
                {"context": ["intact"]},
                {"marker_pos_rel": [0]}]

    # loop over data and average over time-points
    dats_ = {}

    for l in range(len(dataframes)):

        dat_, _ = filter_and_aggregate(dataframes[l], 
                                        model="gpt2", 
                                        model_id=dataframes[l].model_id.unique().item(), 
                                        groups=variables, 
                                        aggregating_metric="mean")
        
        # reformat to zero-padded label (will be useful for sorting)
        dats_[dataframes[l].model_id.unique().item()] = np.median(dat_.x_perc.to_numpy())

    return dats_


# %%
def get_unique_labels(labels):

    unique_heads = list(set([e for l in labels for e in l.split("-")]))
    unique_heads.sort()

    return unique_heads


def populate_data_matrix(data_dict):

    dat = np.zeros(shape=(20, 20))

    unq = get_unique_labels(data_dict.keys())        # find unique labels (i.e. labels of top20 heads)
    label2id = {key: i for i, key in enumerate(unq)}

    # fill the rows and columns of the ablation matrix
    for pairlabel in data_dict.keys():

        r = label2id[pairlabel.split("-")[0]]
        c = label2id[pairlabel.split("-")[1]]
        dat[r, c] = np.round(data_dict[pairlabel], 2)

    return dat


# %%
def plot_imshow(ax, data, labels, vmin=0, vmax=100, cmap=plt.cm.Greys):

    im1 = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)

    # find coordinates of minimum value
    minrow, mincol = np.where(np.tril(data) == np.min(data.flatten()[data.flatten().nonzero()]))
    maxrow, maxcol = np.where(np.tril(data) == np.max(data.flatten()[data.flatten().nonzero()]))

    for xy in [(mincol.item(), minrow.item()), (maxcol.item(), maxrow.item())]:
        x, y = xy
        x -= 0.5
        y -= 0.5
        print(xy)
        rect1 = patches.Rectangle((x, y), 1, 1, linewidth=1.3, edgecolor="red", facecolor="none")
        ax.add_patch(rect1)

    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels(labels, rotation=90)

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    return ax, im1

def save_png_pdf(fig, savename: str):

    savefn = os.path.join(savename + ".png")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, dpi=300, format="png")

    savefn = os.path.join(savename + ".pdf")
    logging.info(f"Saving {savefn}")
    fig.savefig(savefn, format="pdf", transparent=True, bbox_inches="tight")

    return 0


def generate_plot(datadir, which):

    if which == "matching":
        pairs_file = "matching-20-pairs.json"
    elif which == "postmatching":
        pairs_file = "postmatching-20-pairs.json"
    elif which == "recent":
        pairs_file = "recent-20-pairs.json"

    # matching heads pairs
    with open(os.path.join(datadir, pairs_file)) as fh:
        pairs_matching = json.load(fh)

    dfs1 = load_files(datadir, pairlabels=pairs_matching, which=which)

    data_dict1 = get_repeat_surprisals(dataframes=dfs1)

    img_labels = get_unique_labels(data_dict1.keys())


# %%
def main(input_args=None):
    
    datadir = os.path.join(PATHS.data, "ablation\\zero_attn\\topk\\pairs")
    savedir = "C:\\users\\karmeni1\\project\\lm-mem\\fig\\ablation\\topk"

    # matching heads pairs
    with open(os.path.join(datadir, "matching-top20-pairs.json")) as fh:
        pairs_matching = json.load(fh)

    dfs1 = load_files(datadir, pairlabels=pairs_matching, which="matching")

    # postmatching heads pairs
    with open(os.path.join(datadir, "postmatching-top20-pairs.json")) as fh:
        pairs_postmatching = json.load(fh)

    dfs2 = load_files(datadir, pairlabels=pairs_postmatching, which="postmatching")

    # recent tokens heads pairs
    with open(os.path.join(datadir, "recent-top20-pairs.json")) as fh:
        pairs_recent = json.load(fh)

    dfs3 = load_files(datadir, pairlabels=pairs_recent, which="recent")


    data_dict1 = get_repeat_surprisals(dataframes=dfs1)
    data_dict2 = get_repeat_surprisals(dataframes=dfs2)
    data_dict3 = get_repeat_surprisals(dataframes=dfs3)

    matching_labels = get_unique_labels(data_dict1.keys())
    postmatching_labels = get_unique_labels(data_dict2.keys())
    recent_labels = get_unique_labels(data_dict3.keys())

    data1 = populate_data_matrix(data_dict=data_dict1)
    data2 = populate_data_matrix(data_dict=data_dict2)
    data3 = populate_data_matrix(data_dict=data_dict3)

    titlefs = 14
    figsize=(10, 4)
    vmax = np.max([np.max(d) for d in (data1, data2, data3)])
    vmin = 0

    def reformat_label(instr):
        h = int(instr.split("H")[-1]) + 1
        l = int(instr.split("H")[0].strip("L")) + 1
        return f"L{l}.H{h}"

    get_min_max = lambda x: f"Min = {np.round(np.min(x), 1)}%, Max = {np.round(np.max(x),1)}%"

    fig1, ax = plt.subplots(1, 3, figsize=(12, 5.5))
    ax1, im1 = plot_imshow(ax[0], data1, [reformat_label(e) for e in matching_labels], vmin=vmin, vmax=vmax, cmap=plt.cm.viridis)

    minmaxstr = get_min_max(data1.flatten()[data1.flatten().nonzero()])
    #rect1 = patches.Rectangle((0, 19), 1, 1, linewidth=1.3, edgecolor="red", facecolor="red")
    #ax1.add_patch(rect1)
    ax1.text(x=10, y=21, s=minmaxstr, ha="center")

    #ax1.legend(handles=ax1.patches, labels=(minstr, maxstr), bbox_to_anchor=(0, 0), loc='lower left', ncol=2)

    ax1.set_title("Matching heads", color="#2ca02c", fontweight="bold", fontsize=titlefs)
    #cax = ax1.inset_axes([1.03, 0, 0.03, 1])
    #cbar = fig1.colorbar(im1, cax=cax)
    #cbar.ax.set_ylabel("Median repeat surprisal (%)")

    #plt.tight_layout()

    ax2, im2 = plot_imshow(ax[1], data2, [reformat_label(e) for e in postmatching_labels], vmin=vmin, vmax=vmax, cmap=plt.cm.viridis)
    
    minmaxstr = get_min_max(data2.flatten()[data1.flatten().nonzero()])
    ax2.text(x=10, y=21, s=minmaxstr, ha="center")

    ax2.set_title("Post-matching heads", color="#1f77b4", fontweight="bold", fontsize=titlefs)
    #cax = ax2.inset_axes([1.03, 0, 0.03, 1])
    #cbar = fig2.colorbar(im2, cax=cax)
    #cbar.ax.set_ylabel("Median repeat surprisal (%)")

    #plt.tight_layout()

    #fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
    ax3, im3 = plot_imshow(ax[2], data3, [reformat_label(e) for e in recent_labels], vmin=vmin, vmax=vmax, cmap=plt.cm.viridis)

    minmaxstr = get_min_max(data3.flatten()[data1.flatten().nonzero()])
    ax3.text(x=10, y=21, s=minmaxstr, ha="center")

    ax3.set_title("Recent-tokens heads", color="#ff7f0e", fontweight="bold", fontsize=titlefs)
    cax = ax3.inset_axes([1.05, 0, 0.03, 1])
    cbar = fig1.colorbar(im3, cax=cax)
    cbar.ax.set_ylabel("Median repeat surprisal (%)", fontsize=13)

    for a in (ax1, ax2, ax3):
        a.set_xlabel("Top-20 heads", fontsize=13)
        a.xaxis.set_label_position('top')

    ax1.set_ylabel("Top-20 heads", fontsize=13)

    ax[2].annotate('', xy=(26, 2.5), xytext=(26, 17.5),                   
                   arrowprops=dict(arrowstyle='->', color="tab:gray"),  
                   annotation_clip=False)

    ax[2].text(x=26.5, y=10, s='Worse memory', fontsize=12, color="dimgray", rotation=90, va="center")

    plt.tight_layout()
    plt.show()


    if savedir:
        save_png_pdf(fig1, os.path.join(savedir, "paired_ablations"))
        #save_png_pdf(fig1, os.path.join(savedir, "matching_pairs"))
        #save_png_pdf(fig2, os.path.join(savedir, "postmatching_pairs"))
        #save_png_pdf(fig3, os.path.join(savedir, "recenttokens_pairs"))

    plt.show()

    return 0

# %%
if __name__ == "__main__":

    main()