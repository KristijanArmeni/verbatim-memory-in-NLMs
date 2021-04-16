#!/usr/bin/env python
# coding: utf-8

# %%:

import sys, os
sys.path.append(os.path.abspath("C:/users/karmeni1/project/lm-mem/"))
sys.path.append(os.path.abspath("C:/users/karmeni1/project/lm-mem/src/data"))
sys.path.append(os.path.abspath("C:/users/karmeni1/project/lm-mem/src/output"))
sys.path.append(os.path.abspath("C:/users/karmeni1/project/lm-mem/src"))
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib


# set some matplotlib options to handle text better
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# set some seaborn defaults
#sns.set(font_scale=1.3)
sns.set_style("white")  # use the simple white style

# %%

# set paths
home_dir = os.path.join(os.environ['homepath'], "project", "lm-mem", "src")
savedir = os.path.join(os.environ['homepath'], "project", "lm-mem", "fig")
savefigs = True

# In[4]:

data_rnn = pd.read_csv(os.path.join(home_dir, "output", "output_rnn_2.csv"), sep="\t", index_col=0)
data_rnn.rename(columns={"word":"token"}, inplace=True)

data_gpt = pd.read_csv(os.path.join(home_dir, "output", "output_gpt2_2.csv"), sep="\t", index_col=0)
data_gpt.sentid -= 1  # switch to zero indexing

data_gpt["model"] = "gpt-2"
data_rnn["model"] = "lstm"

# %% ===== DESIGN FIGURES ===== %%

# set display and saving dimensions
w, h, w_disp, h_disp = 7, 0.6, 17, 3

# select apropriate conditions
sel = (data_gpt.prompt_len == 8) & \
      (data_gpt.list_len == 10) & \
      (data_gpt.second_list == "repeat") & \
      (data_gpt.context=="intact") & \
      (data_gpt.list=="random") & \
      (data_gpt.sentid.isin([40, 41])) & \
      (data_gpt.model_id=="a-10") & \
      (~data_gpt.token.isin([" ", "<|endoftext|>"]))      

# select data
data = data_gpt.loc[sel].copy()

# %%

# extract single sentence data
x = data.loc[data.sentid==41].reset_index().index.values
y = data.loc[data.sentid==41].surp.to_numpy()
l = data.loc[data.sentid==41].token.to_numpy()

# %%

f, a = plt.subplots(figsize=(w_disp, 2))
a.plot(x, y, marker=".", markerfacecolor="white", linestyle="--", linewidth=1)
#for i, t in enumerate(data.loc[data.sentid==41].token):
#    plt.annotate(text=t, xy=(x[i], y[i]))
a.vlines(x=np.where(l==" soldier"), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
a.vlines(x=np.where(l==" ocean"), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
a.set_xticks(x);
a.set_xticklabels(l, rotation=40, fontsize=6, ha="right");

bluewords = data.loc[data.sentid==41].marker.isin([1, 3]).to_numpy()

[t.set_color("tab:blue") for i, t in enumerate(a.xaxis.get_ticklabels()) if bluewords[i]]

a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)
a.set(ylabel="surprisal", title="Word-by-word surprisal (-LL) GPT-2");

# save
if savefigs:
    f.set_size_inches(w=w, h=h)
    f.savefig(os.path.join(savedir, "example_gpt2.pdf"), transparent=True, dpi=300, bbox_inches="tight")
    f.savefig(os.path.join(savedir, "example_gpt2.png"), transparent=True, dpi=300, bbox_inches="tight")

# %% Example RNN timecourse

sel = (data_rnn.prompt_len==8) &\
      (data_rnn.list_len==10) & \
      (data_rnn.second_list=="repeat") & \
      (data_rnn.context=="intact") & \
      (data_rnn.list=="random") & \
      (data_rnn.model_id=="a-10") & \
      (data_rnn.sentid.isin([40, 41]))

data = data_rnn.loc[sel].copy()

# %% 

x = data.loc[data.sentid==41].reset_index().index.values
y = data.loc[data.sentid==41].surp.to_numpy()
l = data.loc[data.sentid==41].token.to_numpy()

# %% Plot

f, a = plt.subplots(figsize=(w_disp, 2))
a.plot(x, y, marker=".", markerfacecolor="white", linestyle="--", linewidth=1)
#for i, t in enumerate(data.loc[data.sentid==41].token):
#    plt.annotate(text=t, xy=(x[i], y[i]))
a.vlines(x=np.where(l=="soldier"), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
a.vlines(x=np.where(l=="ocean"), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
a.set_xticks(x);
a.set_xticklabels(l, rotation=40, fontsize=6, ha="right");

bluewords = data.loc[data.sentid==41].marker.isin([1, 3]).to_numpy()

[t.set_color("tab:blue") for i, t in enumerate(a.xaxis.get_ticklabels()) if bluewords[i]]

a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)

a.set(ylabel="surprisal", title="Word-by-word surprisal (-LL) LSTM");

# %% save

if savefigs:
    f.set_size_inches(w=w, h=h)
    f.savefig(os.path.join(savedir, "./example_rnn.pdf"), transparent=True, dpi=300, bbox_inches="tight")
    f.savefig(os.path.join(savedir, "./example_rnn.png"), transparent=True, dpi=300, bbox_inches="tight")

# %% ===== MAIN FIGURES ===== %%

# Experiment 1 and 2: word order and semantic structure

# set figure dimensions
w, h, w_disp, h_disp = 10, 1.5, 17, 3

data = pd.concat([data_gpt, data_rnn])

# rename row values 
data.loc[data["list"]=="categorized", "list"] = "semantic"
data.loc[data["list"]=="random", "list"] = "arbitrary"
data.loc[data["second_list"]=="control", "second_list"] = "novel"
data.loc[data["second_list"]=="repeat", "second_list"] = "repeated"
data.loc[data["second_list"]=="permute", "second_list"] = "permuted"

# In[23]:

context_len = 8         # select short context
list_len = 10           # select shortest list
context = "intact"      # take the intact context
markers = [2, 3]
marker_range = list(range(-4, 10))
list_types = ["arbitrary", "semantic"]

sel = (data.prompt_len == context_len) & \
      (data.list_len == list_len) & \
      (data.list.isin(list_types)) & \
      (data.context == context) & \
      (data.model_id == "a-10") & \
      (data.marker.isin(markers)) & \
      (data.marker_pos_rel.isin(marker_range))
      
d = data.loc[sel].copy()

# name column manually
d.rename(columns={"list": "list structure", "second_list": "condition"}, inplace=True)


# %% wrapper function


def plot_timecourse(data, nouns):
    
    p = sns.relplot(kind="line", data=data, x="marker_pos_rel", y="surp", style="condition", col="model", 
                    estimator=np.mean, ci=95.0, err_style="bars",
                    markers=True, style_order=["repeated", "permuted", "novel"],
                    legend="auto", linewidth=1)

    p.fig.subplots_adjust(top=0.85)
    p.set_titles(col_template="{col_name}")
    p.set_axis_labels("token position in list" , "mean surprisal\n[95% ci]")

    p.fig.suptitle("LM word list ({} nouns) recognition memory".format(nouns))


    p.set(xticks=list(range(-4,10)))
    #p.set_xticklabels(rotation=20)
    p.despine(left=True);
    return p


# %% plot arbitrary nouns first

nouns = "arbitrary"
p = plot_timecourse(d.loc[d["list structure"] == nouns], nouns=nouns)
p.fig.set_size_inches(w=w_disp, h=h_disp)

# save
if savefigs:
    p.fig.set_size_inches(w=w, h=h)
    p.savefig(os.path.join(savedir, "word_order_{}_nouns.pdf".format(nouns)), transparent=True, bbox_inches="tight")
    p.savefig(os.path.join(savedir, "word_order_{}_nouns.png".format(nouns)), dpi=300, transparent=True, bbox_inches="tight")

# In[47]:

nouns="semantic"
p = plot_timecourse(d.loc[d["list structure"] == nouns], nouns=nouns)
p.fig.set_size_inches(w=w_disp, h=h_disp)

# save
if savefigs:
    p.fig.set_size_inches(w=w, h=h)
    p.savefig(os.path.join(savedir, "word_order_{}_nouns.pdf".format(nouns)), transparent=True, bbox_inches="tight")
    p.savefig(os.path.join(savedir, "word_order_{}_nouns.png".format(nouns)), dpi=300, transparent=True, bbox_inches="tight")

# %% ======================= %%
# ======== LIST LENGTH ===== %%

data = None
data = pd.concat([data_gpt, data_rnn])

# select repeat condition and all list lengths
context_len = 8
list_len = [3, 5, 10]
context = "intact"

# we drop the first token
sel = (data.prompt_len==context_len) &\
      (data.list_len.isin(list_len)) &\
      (data.second_list.isin(["repeat", "permute", "control"])) & \
      (~data.token.isin(["<|endoftext|>", " "])) & \
      (data.list.isin(["random", "categorized"])) & \
      (data.context==context) & \
      (data.model_id=="a-10") & \
      (data.marker.isin([1, 3])) & \
      (data.marker_pos_rel.isin(list(range(1, 10))))

d = data.loc[sel].copy()

# Now, we average over tokens
units = ["list_len", "sentid", "model", "marker", "list", "second_list"]
dagg = d.groupby(units).agg({"surp": ["mean", "std"]}, {"token": list}).reset_index()
dagg.columns = ['_'.join(col_v) if col_v[-1] != '' else col_v[0] for col_v in dagg.columns.values]


# %% 
# Define function that computes relative change in average surprisal
def get_relative_change(x1=None, x2=None, labels1=None, labels2=None):
    
    """
    computes relative change across data in x1 and x2. Sizes of arrays x1 and x2
    should match.
    """
    
    # check that labels match
    if (labels1 is not None) & (labels2 is not None):
        assert (labels1 == labels2).all()
    
    x_del = ((x2-x1)/(x1+x2))*100
    
    return x_del


# %%

def make_bar_plot(data_frame, x, y, hue, col,
                  xlabel=None, ylabel=None, suptitle=None,
                  size_inches=(5, 3), legend_title=None, hue_order=["unseen", "permuted", "repeated"]):


    g = sns.catplot(data=data_frame, x=x, y=y, hue=hue, col=col, 
                    kind="bar", dodge=0.5, palette="tab10", zorder=2, legend=False, seed=12345,
                    hue_order=hue_order,
                    facecolor=(1, 1, 1, 0), edgecolor=["tab:gray"], ecolor=["tab:gray"], bottom=0)
    
    ax = g.axes[0]

    # left panel
    select = (data_frame.model=="gpt-2")
    sns.stripplot(ax=ax[0], data=data_frame[select], x=x, y=y, hue=hue, hue_order=hue_order,
                  palette="tab10", dodge=0.5, alpha=0.25, zorder=1)


    # right panel
    select = (data_frame.model=="lstm")
    sns.stripplot(ax=ax[1], data=data_frame[select], x=x, y=y, hue=hue, hue_order=hue_order,
                  palette="tab10", dodge=0.5, alpha=0.25, zorder=1)

    # set labels
    ax[0].set_ylabel(ylabel)
    ax[1].set_ylabel("")
    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(xlabel)
    
    # make bars darker than pointclouds for visual benefit
    blue, orange, green = sns.color_palette("dark")[0:3]
    n_x_groups = len(data_frame[x].unique())
    for i in [0, 1]:
        for patch in ax[i].patches[0:n_x_groups]:
            patch.set_edgecolor(blue)
        for patch in ax[i].patches[n_x_groups:n_x_groups*2]:
            patch.set_edgecolor(orange)
        for patch in ax[i].patches[n_x_groups*2::]:
            patch.set_edgecolor(green)
    
    # annotate
    x_levels = data_frame.loc[x].unique()
    col_levels = data_frame.loc[col].unique()
    hue_levels = data_frame.loc[hue].unique()
    
    # find n rows for one plotted group
    one_group = (data_frame[x]==x_levels[0]) & (data_frame[hue]==hue_levels[0]) & (data_frame[col] == col_levels[0])
    n = len(data_frame.loc[one_group])  
    
    ax[1].text(x=ax[1].get_xlim()[0], y=ax[1].get_ylim()[0]*0.9, s="N = {}".format(n))
    
    # legend
    # Improve the legend 
    ax[0].get_legend().remove()

    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[0:3], labels[0:3],
              handletextpad=1, columnspacing=1,
              loc="lower right", ncol=1, frameon=False, title="target list")

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("{} ({} nouns)".format(suptitle, list_type))
    g.set_titles(col_template="{col_name}")
    g.despine(left=True)
    g.fig.set_size_inches(size_inches[0], size_inches[1])
    
    return g, ax


# %% compute relative change
# apply relatvie change to each group and apply to 
df_list = []
for model in ["gpt-2", "lstm"]:
    for length in [3, 5, 10]:
        for condition in ["repeat", "permute", "control"]:
            for list_type in ["categorized", "random"]:
            
                cols = ["x1", "x2", "x_del"]
                df = pd.DataFrame(columns=cols)

                select = (dagg.model == model) & (dagg.list_len == length) & (dagg.second_list == condition) & (dagg.list==list_type)
                d = dagg.loc[select].copy()

                x1=d.loc[d.marker==1].surp_mean.to_numpy()           # average per sentence surprisal on first list
                x2=d.loc[d.marker==3].surp_mean.to_numpy()           # average per sentence surprisal on second list
                labels1 = d.loc[d.marker==1].sentid.to_numpy()  # use sentence id for check
                labels2 = d.loc[d.marker==3].sentid.to_numpy()

                x_del = get_relative_change(x1=x1, x2=x2, labels1=labels2, labels2=labels2)

                df["x1"] = x1
                df["x2"] = x2
                df["x_del"] = x_del
                df["model"] = model
                df["list_len"] = length
                df["list"] = list_type
                df["condition"] = condition

                df_list.append(df)


# In[111]:


# concatenate relative scores
data = None
data = pd.concat(df_list)

data.loc[data["list"]=="categorized", "list"] = "semantic"
data.loc[data["list"]=="random", "list"] = "arbitrary"
data.loc[data["condition"]=="control", "condition"] = "unseen"
data.loc[data["condition"]=="repeat", "condition"] = "repeated"
data.loc[data["condition"]=="permute", "condition"] = "permuted"


# In[112]:


list_type="arbitrary"
p1, a = make_bar_plot(data_frame=data.loc[data.list==list_type], 
                  x="list_len", y="x_del", hue="condition", col="model",
                  xlabel="list length (n. tokens)", ylabel="surprisal level on target list\n(% change)", 
                  suptitle="LM recognition memory as change in surprisal", legend_title="target list",
                  size_inches=(5, 3))


# In[113]:


if savefigs:
    p1.savefig(os.path.join(savedir, "list_lentgh_{}_nouns.pdf".format(list_type)), transparent=True, bbox_inches="tight")
    p1.savefig(os.path.join(savedir, "list_length_{}_nouns.png".format(list_type)), dpi=300, transparent=True, bbox_inches="tight")


# In[114]:


list_type="semantic"
p2, _ = make_bar_plot(data_frame=data.loc[data.list==list_type], 
                  x="list_len", y="x_del", hue="condition", col="model",
                  xlabel="list length (n. tokens)", ylabel="surprisal level on target list\n(% change)", 
                  suptitle="LM recognition memory as change in surprisal", legend_title="target list",
                  size_inches=(5, 3))


# In[75]:


if savefigs:
    p2.savefig(os.path.join(savedir, "list_length_{}_nouns.pdf".format(list_type)), transparent=True, bbox_inches="tight")
    p2.savefig(os.path.join(savedir, "list_length_{}_nouns.png".format(list_type)), dpi=300, transparent=True, bbox_inches="tight")


# ## Experiment 4: effect of context length

# In[115]:


data = None
data = pd.concat([data_gpt, data_rnn])


# In[116]:


# select repeat condition and all list lengths
context_len = [8,  30, 100, 200, 400]
list_len = 10
context = "intact"

# we drop the first token
sel = (data.prompt_len.isin(context_len)) &       (data.list_len.isin([list_len])) &       (data.second_list.isin(["repeat", "permute", "control"])) &       (~data.token.isin(["<|endoftext|>", " "])) &       (data.list.isin(["random", "categorized"])) &       (data.context==context) &       (data.model_id=="a-10") &       (data.marker.isin([1, 3])) &       (data.marker_pos_rel.isin(list(range(1, 10))))

d = data.loc[sel].copy()


# In[117]:


units = ["prompt_len", "sentid", "model", "marker", "list", "second_list"]
dagg = d.groupby(units).agg({"surp": ["mean", "std"]}, {"token": list}).reset_index()
dagg.columns = ['_'.join(col_v) if col_v[-1] != '' else col_v[0] for col_v in dagg.columns.values]


# In[118]:


# apply relative change computation and apply to 

df_list = []
for model in ["gpt-2", "lstm"]:
    for length in [8,  30, 100, 200, 400]:
        for condition in ["repeat", "permute", "control"]:
            for list_type in ["categorized", "random"]:
            
                cols = ["x1", "x2", "x_del"]
                df = pd.DataFrame(columns=cols)

                select = (dagg.model == model) & (dagg.prompt_len == length) & (dagg.second_list == condition) & (dagg.list==list_type)
                d = dagg.loc[select].copy()

                x1=d.loc[d.marker==1].surp_mean.to_numpy()           # average per sentence surprisal on first list
                x2=d.loc[d.marker==3].surp_mean.to_numpy()           # average per sentence surprisal on second list
                labels1 = d.loc[d.marker==1].sentid.to_numpy()  # use sentence id for check
                labels2 = d.loc[d.marker==3].sentid.to_numpy()

                x_del = get_relative_change(x1=x1, x2=x2, labels1=labels2, labels2=labels2)

                df["x1"] = x1
                df["x2"] = x2
                df["x_del"] = x_del
                df["model"] = model
                df["prompt_len"] = length
                df["list"] = list_type
                df["condition"] = condition

                df_list.append(df)


# In[119]:


# concatenate relative scores
data = None
data = pd.concat(df_list)


# In[120]:


data.loc[data["list"]=="categorized", "list"] = "semantic"
data.loc[data["list"]=="random", "list"] = "arbitrary"
data.loc[data["condition"]=="control", "condition"] = "unseen"
data.loc[data["condition"]=="repeat", "condition"] = "repeated"
data.loc[data["condition"]=="permute", "condition"] = "permuted"


# In[121]:


list_type="arbitrary"
p3, _ = make_bar_plot(data_frame=data.loc[data.list==list_type], 
                      x="prompt_len", y="x_del", hue="condition", col="model",
                      xlabel="context length (n. tokens)", ylabel="surprisal level on target list\n(% change)", 
                      suptitle="LM recognition memory as change in surprisal", legend_title="target list",
                      size_inches=(6, 3))

if savefigs:
    p3.savefig(os.path.join(savedir, "context_lentgh_{}_nouns.pdf".format(list_type)), transparent=True, bbox_inches="tight")
    p3.savefig(os.path.join(savedir, "context_length_{}_nouns.png".format(list_type)), dpi=300, transparent=True, bbox_inches="tight")

list_type="semantic"
p4, _ = make_bar_plot(data_frame=data.loc[data.list==list_type], 
                      x="prompt_len", y="x_del", hue="condition", col="model",
                      xlabel="context length (n. tokens)", ylabel="surprisal level on target list\n(% change)", 
                      suptitle="LM recognition memory as change in surprisal", legend_title="target list",
                      size_inches=(6, 3))

if savefigs:
    p4.savefig(os.path.join(savedir, "context_lentgh_{}_nouns.pdf".format(list_type)), transparent=True, bbox_inches="tight")
    p4.savefig(os.path.join(savedir, "context_length_{}_nouns.png".format(list_type)), dpi=300,  transparent=True, bbox_inches="tight")


# ## Experiment 5: effect of short context

# In[133]:


def make_bar_plot2(data_frame, x, y, hue,
                   xlabel=None, ylabel=None, suptitle=None,
                   size_inches=(5, 3), legend_title=None, hue_order=["unseen", "permuted", "repeated"]):


    g = sns.catplot(data=data_frame, x=x, y=y, hue=hue,
                    kind="bar", dodge=0.5, palette="tab10", zorder=2, legend=False, seed=12345,
                    hue_order=hue_order,
                    facecolor=(1, 1, 1, 0), edgecolor=["tab:gray"], ecolor=["tab:gray"], bottom=0)
    ax = g.axes[0]

    # right panel
    #select = (data.model=="gpt-2")
    #sns.stripplot(ax=ax[0], data=data[select], x=x, y=y, hue=hue, hue_order=hue_order,
    #              palette="tab10", dodge=0.5, alpha=0.25, zorder=1)


    # right panel
    select = (data_frame.model=="lstm")
    sns.stripplot(ax=ax[0], data=data_frame[select], x=x, y=y, hue=hue, hue_order=hue_order,
                  palette="tab10", dodge=0.5, alpha=0.25, zorder=1)

    # set labels
    ax[0].set_ylabel(ylabel)
    #ax[1].set_ylabel("")
    ax[0].set_xlabel(xlabel)
    #ax[1].set_xlabel(xlabel)
    
    blue, orange, green = sns.color_palette("dark")[0:3]
    n_x_groups = len(data_frame[x].unique())
    for i in [0]:
        for patch in ax[i].patches[0:n_x_groups]:
            patch.set_edgecolor(blue)
        for patch in ax[i].patches[n_x_groups:n_x_groups*2]:
            patch.set_edgecolor(orange)
        for patch in ax[i].patches[n_x_groups*2::]:
            patch.set_edgecolor(green)

    # legend
    # Improve the legend 
    #ax[0].get_legend().remove()

    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles[0:3], labels[0:3],
              handletextpad=1, columnspacing=1,
              loc="upper left", bbox_to_anchor=(1, 1), ncol=1, frameon=False, title="target list")

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("{} ({} nouns)".format(suptitle, list_type))
    g.set_titles(col_template="{col_name}")
    g.despine(left=True)
    g.fig.set_size_inches(size_inches[0], size_inches[1])
    
    return g, ax


# In[134]:


data = None
data = data_rnn


# In[135]:


# select repeat condition and all list lengths
context_len = 8
list_len = [5]
context = "short"

# we drop the first token
sel = (data.prompt_len==context_len) &       (data.list_len.isin(list_len)) &       (data.second_list.isin(["repeat", "permute", "control"])) &       (~data.token.isin(["<|endoftext|>", " "])) &       (data.list.isin(["random"])) &       (data.context==context) &       (data.model_id=="a-10") &       (data.marker.isin([0, 1, 2, 3]))

d = data.loc[sel].copy()


# ## Plot a time course

# In[136]:


sent_id=21
second_list="repeat"
x = d.loc[(d.sentid==sent_id) & (d.second_list==second_list)].reset_index().index.values
y = d.loc[(d.sentid==sent_id) & (d.second_list==second_list)].surp.to_numpy()
l = d.loc[(d.sentid==sent_id) & (d.second_list==second_list)].token.to_numpy()


# In[137]:


f, a = plt.subplots(figsize=(w_disp, 2))
a.plot(x, y, marker=".", markerfacecolor="white", linestyle="--", linewidth=1)

a.vlines(x=np.where(l=="mixture"), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
a.vlines(x=np.where(l=="berry"), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
a.set_xticks(x);
a.set_xticklabels(l, rotation=40, fontsize=6, ha="right");

bluewords = d.loc[(d.sentid==sent_id) & (d.second_list==second_list)].marker.isin([1, 3]).to_numpy()

[t.set_color("tab:blue") for i, t in enumerate(a.xaxis.get_ticklabels()) if bluewords[i]]

a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)

a.set(ylabel="surprisal", title="Word-by-word surprisal (-LL) LSTM");


# In[138]:


if savefigs:
    f.set_size_inches(w, h)
    f.savefig(os.path.join(savedir, "example_short_context_{}_nouns.pdf".format(list_type)), transparent=True, bbox_inches="tight")


# In[139]:


units = ["list_len", "sentid", "model", "marker", "list", "second_list"]
dagg = d.groupby(units).agg({"surp": ["median", "std"]}, {"token": list}).reset_index()
dagg.columns = ['_'.join(col_v) if col_v[-1] != '' else col_v[0] for col_v in dagg.columns.values]


# In[140]:


# apply relative change computation and apply to 

df_list = []
for model in ["lstm"]:
    for length in [3,  5, 10]:
        for condition in ["repeat", "permute", "control"]:
            for list_type in ["categorized", "random"]:
            
                cols = ["x1", "x2", "x_del"]
                df = pd.DataFrame(columns=cols)

                select = (dagg.model == model) & (dagg.list_len == length) & (dagg.second_list == condition) & (dagg.list==list_type)
                d = dagg.loc[select].copy()

                x1=d.loc[d.marker==1].surp_median.to_numpy()           # average per sentence surprisal on first list
                x2=d.loc[d.marker==3].surp_median.to_numpy()           # average per sentence surprisal on second list
                labels1 = d.loc[d.marker==1].sentid.to_numpy()  # use sentence id for check
                labels2 = d.loc[d.marker==3].sentid.to_numpy()

                x_del = get_relative_change(x1=x1, x2=x2, labels1=labels2, labels2=labels2)

                df["x1"] = x1
                df["x2"] = x2
                df["x_del"] = x_del
                df["model"] = model
                df["list_len"] = length
                df["list"] = list_type
                df["condition"] = condition

                df_list.append(df)


# In[141]:


# concatenate relative scores
data = None
data = pd.concat(df_list)


# In[142]:


data.loc[data["list"]=="categorized", "list"] = "semantic"
data.loc[data["list"]=="random", "list"] = "arbitrary"
data.loc[data["condition"]=="control", "condition"] = "unseen"
data.loc[data["condition"]=="repeat", "condition"] = "repeated"
data.loc[data["condition"]=="permute", "condition"] = "permuted"


# In[143]:


list_type="arbitrary"
p5, _ = make_bar_plot2(data_frame=data.loc[data.list==list_type], 
                      x="list_len", y="x_del", hue="condition",
                      xlabel="list length (n. tokens)", ylabel="surprisal level on target list\n(% change)", 
                      suptitle="LM recognition memory as change in surprisal", legend_title="target list",
                      size_inches=(2, 4))

if savefigs:
    p5.savefig(os.path.join(savedir, "short_context_{}_nouns.pdf".format(list_type)), transparent=True, bbox_inches="tight")
    p5.savefig(os.path.join(savedir, "short_context_{}_nouns.png".format(list_type)), dpi=300, transparent=True, bbox_inches="tight")

list_type="semantic"
p6, _ = make_bar_plot2(data_frame=data.loc[data.list==list_type], 
                      x="list_len", y="x_del", hue="condition",
                      xlabel="list length (n. tokens)", ylabel="surprisal level on target list\n(% change)", 
                      suptitle="LM recognition memory as change in surprisal", legend_title="target list",
                      size_inches=(2, 4))

if savefigs:
    p6.savefig(os.path.join(savedir, "short_context_{}_nouns.pdf".format(list_type)), transparent=True, bbox_inches="tight")
    p6.savefig(os.path.join(savedir, "short_context_{}_nouns.png".format(list_type)), dpi=300, transparent=True, bbox_inches="tight")


# ## Experiment 6: n-gram experiment

# In[ ]:


# select appropriate rows
selection = (data_rnn.list=="ngram-random") & (data_rnn.marker.isin([1, 2])) & (data_rnn.model_id=="a-10")
rnn = data_rnn.loc[selection].copy()


# In[ ]:


selection = (data_gpt.list=="ngram-random") & (data_gpt.marker==1) & (~data_gpt.token.isin(["<|endoftext|>", " "]))
gpt = data_gpt.loc[selection].copy()


# In[ ]:


# drop columns we don't need here
sel = ["prompt_len", "list_len", "context", "second_list"]
rnn.drop(columns=sel+["hs", "dHs"], inplace=True)
#gpt.drop(columns=sel+["prefix"], inplace=True)


# ### Count n-gram positions (for x-axis)

# In[ ]:


# let's create a ngram position counter
def add_ngram_columns(data_frame):
    
    df= None
    df = data_frame.copy()
    
    tmp = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    
    for ind in df.sentid.unique():

        sel = df.loc[df.sentid == ind]
        
        listlen = len(sel)
        ngram_ = int(sel.ngram_len.iloc[0])
        dist_ = int(sel.dist_len.iloc[0])
        ngram = int(listlen/5)
        npositions = int(listlen/ngram)
        ngram_pos = np.repeat(np.arange(0, 5), ngram_+dist_)
        if dist_ != 0:
            ngram_pos = ngram_pos[:-dist_]
            
        tmp.append(ngram_pos) # index ngram position withing sequence
        #tmp2.append(np.full(listlen, ngram))                   # index ngram lenght
        #tmp3.append(np.tile(np.arange(0, ngram), npositions))  # index tokens
        #tmp4.append(np.tile(np.repeat((ngram != ngram_), ngram), npositions))
        
    df["ngram_pos"] = np.concatenate(tmp)
    #df["ngram_size"] = np.concatenate(tmp2)
    #df["token_id"] = np.concatenate(tmp3)
    #df["has_subtokens"] = np.concatenate(tmp4)
    
    return df


# In[ ]:


rnn = add_ngram_columns(rnn)


# In[ ]:


gpt = add_ngram_columns(gpt)


# In[ ]:


rnn.ngram_pos += 1
#gpt.ngram_pos += 1
#rnn.token_id += 1
#gpt.token_id += 1


# In[ ]:


# enforce 2, 3, and 5 grams for gpt-2, even if BPE (drop if longer)
indexNames = gpt.loc[(gpt.has_subtokens) & (gpt.ngram_size == 4) & (gpt.token_id==4)].index
gpt.drop(indexNames, axis=0, inplace=True)
indexNames = gpt.loc[(gpt.has_subtokens) & (gpt.ngram_size == 6) & (gpt.token_id==6)].index
gpt.drop(indexNames, axis=0, inplace=True)
         
# then rename the truncated rows accordingly
gpt.loc[gpt.ngram_size==4, "ngram_size"] = 3
gpt.loc[gpt.ngram_size==6, "ngram_size"] = 5


# ### Plot some trial trime courses for n-grams 

# In[ ]:


def plot_trial(data, sentence_id, ylabel, title=None, size_inches=(10, 2)):
    
    x = data.loc[data.sentid==sentence_id].reset_index().index.values
    y = data.loc[data.sentid==sentence_id].surp.to_numpy()
    l = data.loc[data.sentid==sentence_id].token.to_numpy()
    groups = data.loc[data.sentid==sentence_id].ngram_pos
    
    f, a = plt.subplots(figsize=size_inches)
    
    a.plot(x, y, marker=".", markerfacecolor="white", linestyle="--", linewidth=1)

    a.vlines(x=np.where(l==l[0]), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
    a.set_xticks(x);
    a.set_xticklabels(l, rotation=40, fontsize=10, ha="center");

    bluewords = groups.isin([1, 3, 5]).to_numpy()

    [t.set_color("tab:blue") for i, t in enumerate(a.xaxis.get_ticklabels()) if bluewords[i]]

    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

    a.set(ylabel=ylabel, title=title);
    
    return f, a


# In[ ]:


def plot_trial2(data, sentence_id, ylabel, title=None, size_inches=(10, 2)):
    
    x = data.loc[data.sentid==sentence_id].reset_index().index.values
    y = data.loc[data.sentid==sentence_id].surp.to_numpy()
    l = data.loc[data.sentid==sentence_id].token.to_numpy()
    groups = data.loc[data.sentid==sentence_id].marker
    
    last_tok = data.loc[(data.sentid==sentence_id) & (data.marker == 1) & (data.ngram_pos == 0)].token.to_list()[-1]
    
    f, a = plt.subplots(figsize=size_inches)
    
    a.plot(x, y, marker=".", markerfacecolor="white", linestyle="--", linewidth=1)

    a.vlines(x=np.where(l==l[0]), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
    a.vlines(x=np.where(l==last_tok), ymin=a.get_ylim()[0], ymax=a.get_ylim()[-1], color="r", linestyle="--", linewidth=1)
    a.set_xticks(x);
    a.set_xticklabels(l, rotation=40, fontsize=12, ha="center");

    bluewords = groups.isin([1]).to_numpy()

    [t.set_color("tab:blue") for i, t in enumerate(a.xaxis.get_ticklabels()) if bluewords[i]]

    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

    a.set(ylabel=ylabel, title=title);
    
    return f, a


# In[ ]:


data=None
data=rnn
f1, a1 = plot_trial(data=data, sentence_id=10, ylabel="surprisal", title="RNN suprisal (repeated 2-grams)", size_inches=(7, 0.8))
f2, a2 = plot_trial(data=data, sentence_id=21, ylabel="surprisal", title="RNN suprisal (repeated 3-grams)", size_inches=(7, 0.8))
f3, a3 = plot_trial(data=data, sentence_id=60, ylabel="surprisal", title="RNN suprisal (repeated 5-grams)", size_inches=(7, 0.8))
#f1.savefig(os.path.join(savedir, "2gram_example_RNN_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")
#f2.savefig(os.path.join(savedir, "3gram_example_RNN_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")
#f3.savefig(os.path.join(savedir, "5gram_example_RNN_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")


# In[ ]:


data=None
data=gpt
f1, a1 = plot_trial(data=data, sentence_id=10, ylabel="surprisal", title="GPT-2 suprisal (repeated 2-grams)", size_inches=(7, 0.8))
f2, a2 = plot_trial(data=data, sentence_id=21, ylabel="surprisal", title="GPT-2 suprisal (repeated 3-grams)", size_inches=(7, 0.8))
f3, a3 = plot_trial(data=data, sentence_id=41, ylabel="surprisal", title="GPT-2 suprisal (repeated 5-grams)", size_inches=(7, 0.8))
f1.savefig(os.path.join(savedir, "2gram_example_GPT-2_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")
f2.savefig(os.path.join(savedir, "3gram_example_GPT-2_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")
f3.savefig(os.path.join(savedir, "5gram_example_GPT-2_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")


# ### Averaging per ngram position

# In[ ]:


f1, a1 = plot_trial2(data=data, sentence_id=30, ylabel="surprisal", title="RNN suprisal", size_inches=(15, 1.5))
f2, a2 = plot_trial2(data=data, sentence_id=60, ylabel="surprisal", title="RNN suprisal", size_inches=(15, 1.5))
f3, a3 = plot_trial2(data=data, sentence_id=180, ylabel="surprisal", title="RNN suprisal", size_inches=(15, 1.5))
f4, a4 = plot_trial2(data=data, sentence_id=300, ylabel="surprisal", title="RNN suprisal", size_inches=(15, 1.5))


# In[ ]:


f1.savefig(os.path.join(savedir, "2gram_example_GPT-2_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")
f2.savefig(os.path.join(savedir, "3gram_example_GPT-2_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")
f3.savefig(os.path.join(savedir, "5gram_example_GPT-2_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")
f4.savefig(os.path.join(savedir, "5gram_example_GPT-2_arbitrary_nouns.pdf"), transparent=True, bbox_inches="tight")


# In[ ]:


#gpt["model"] = "gpt-2"
#rnn["model"] = "rnn"
rnnsel = rnn.loc[rnn.marker == 1].copy()
rnngpt_agg = rnnsel.groupby(["model", "ngram_len", "dist_len", "ngram_pos", "sentid"])                .agg({"surp": ["mean", "std"], "token": list})                .reset_index()
rnngpt_agg.columns = ['_'.join(col_v) if col_v[-1] != '' else col_v[0] for col_v in rnngpt_agg.columns.values]


# In[ ]:


p = sns.catplot(data=rnngpt_agg, kind="point", x="ngram_pos", y="surp_mean", hue="dist_len", col="ngram_len",
                estimator=np.mean, ci=95.0, n_boot=1000, seed=12345,
                legend_out=False, dodge=0.15, sharey=True,
                palette=sns.color_palette("rocket_r"))
p.fig.set_size_inches(20, 4)  
p.fig.subplots_adjust(top=0.75)
p.fig.suptitle("Are immediately repeated word lists lingering in memory?")
p.set_axis_labels( "ngram serial position in list" , "mean surprisal\n(95% ci)")
p._legend.set_title("distractor length")
p.set_titles("ngram length = {col_name}")
p.despine(left=True);


# In[ ]:


p.savefig(os.path.join(savedir, "rnn_ngram.png"), transparent=True, bbox_inches="tight")
p.savefig(os.path.join(savedir, "rnn_ngram.pdf"), transparent=True, bbox_inches="tight")


# ## Averaging per token position within ngram

# In[ ]:


gpt["model"] = "gpt-2"
rnn["model"] = "rnn"
agg = pd.concat([rnn, gpt]).groupby(["model", "prompt_len", "token_id", "sentid", "ngram_size"])                .agg({"surp": ["mean", "std"], "token": list})                .reset_index()
agg.columns = ['_'.join(col_v) if col_v[-1] != '' else col_v[0] for col_v in agg.columns.values]


# In[ ]:


p = sns.catplot(data=agg, kind="point", x="token_id", y="surp_mean", hue="ngram_size", col="model",
                estimator=np.mean, ci=95.0, n_boot=1000, seed=12345,
                legend_out=False, dodge=0.15, sharey=False,
                palette=sns.color_palette("rocket_r"))
p.fig.set_size_inches(6, 2)  
p.fig.subplots_adjust(top=0.75)
p.fig.suptitle("Are early tokens remembered better than late?")
p.set_axis_labels( "token position within ngram" , "mean surprisal\n(95% ci)")
p._legend.set_title("ngram")
p.set_titles("{col_name}")
p.despine(left=True);


# ## Supplementary

# In[ ]:


def make_plot2(data, nouns):
    p = sns.relplot(kind="line", data=data, x="marker_pos_rel", y="surp", hue="model_id", 
                    estimator=np.mean, ci=95.0, err_style="bars",
                    markers=True, style_order=["repeated", "permuted", "unseen"],
                    legend="auto", linewidth=2.5)
    p.fig.set_size_inches(14, 4)
    p.fig.subplots_adjust(top=0.85)
    p.set_titles(col_template="{col_name}")
    p.set_axis_labels("token position in list" , "mean surprisal\n[95% ci]")
    #p._legend.set_title("condition")
    p.fig.suptitle("LM word list ({} nouns) recognition memory".format(nouns))

    # set informative tick labels
    #labs = data.loc[(data.marker==2) & (data.marker_pos_rel.isin(list(range(-4, 0)))) & (data.sentid==160)] \
    #               .token[0:4].tolist() \
    #               + list(range(0, 10))
    #labs = [str(e) for e in labs]

    p.set(xticks=list(range(-4,10)))
    #p.set_xticklabels(rotation=20)
    p.tight_layout()
    p.despine(left=True);
    return p


# In[ ]:


data=None
data=data_rnn
data.loc[data["list"]=="categorized", "list"] = "semantic"
data.loc[data["list"]=="random", "list"] = "arbitrary"
data.loc[data["second_list"]=="control", "second_list"] = "unseen"
data.loc[data["second_list"]=="repeat", "second_list"] = "repeated"
data.loc[data["second_list"]=="permute", "second_list"] = "permuted"


# In[ ]:


sel = (data.prompt_len==100) &       (data.list_len==10) &       (data.list.isin(["arbitrary", "semantic"])) &       (data.context=="intact") &       (data.second_list=="repeated") &       (data.model=="lstm") &       (data.marker.isin([2, 3])) &       (data.marker_pos_rel.isin(list(range(-4, 10))))
d = data.loc[sel].copy()

# name column manually
d.rename(columns={"list": "list structure", "second_list": "condition"}, inplace=True)


# In[ ]:


nouns="arbitrary"
p = make_plot2(d.loc[d["list structure"] == nouns], nouns=nouns)
p.fig.set_size_inches(w=w_disp, h=h_disp)


# In[ ]:


nouns="semantic"
p = make_plot2(d.loc[d["list structure"] == nouns], nouns=nouns)
p.fig.set_size_inches(w=w_disp, h=h_disp)

