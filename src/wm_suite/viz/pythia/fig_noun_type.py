# %%
import json
import os
from itertools import product

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import trim_mean, bootstrap
from wm_suite.paths import ROOT_PATH, DATA_PATH
from wm_suite.viz.utils import save_png_pdf, logger
import seaborn as sns
from tqdm import tqdm

from fig_mem import N_TOKENS_PER_BATCH, plot_lines

# %%
STEPS = [
    "step0",
    "step1",
    "step4",
    "step32",
    "step128",
    "step256",
    "step512",
    "step1000",
    "step2000",
    "step3000",
    "step4000",
    "step8000",
    "step10000",
    "step30000",
    "step40000",
    "step50000",
    "step100000",
    "step143000",
]

# %%
def load_abstract_concrete_data(step:int=143000) -> dict:
    models = ["pythia-14m",
              "pythia-31m",
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
            "pythia-12b"]

    conditions = ["concrete", "abstract"]

    datadir = os.path.join(ROOT_PATH.parent.parent.parent, "data", "abst-conc")

    dftmp = {k: [] for k in ["model", "rs", "cond"]}
    for tup in tqdm(list(product(models, conditions)), desc='checkpoint'):

        model, cond = tup

        fn = os.path.join(datadir, model, f"pythia_{model}_step-{step}_{cond}_repeat_mem.json")

        with open(fn, 'r') as f:
            mem = json.load(f)
        dftmp["model"].append(model.strip("pythia-").upper())
        dftmp["rs"].append(np.array(mem["rs"]))
        dftmp["cond"].append(cond.capitalize())

    return dftmp

# %%
def load_timecourse_data(datadir:str, which:str) -> dict:

    out = {}

    ckp_steps_dict = {
        "pythia-14m": STEPS,
        "pythia-31m": STEPS,
        "pythia-70m": STEPS,
        "pythia-160m": STEPS,
        "pythia-410m": STEPS,
        "pythia-1b": STEPS,
        "pythia-1.4b": STEPS,
        "pythia-2.8b": STEPS,
        "pythia-6.9b": STEPS,
        "pythia-12b": STEPS,
    }

    for ckp, ckp_steps in tqdm(ckp_steps_dict.items(), desc='checkpoint'):

        out[ckp] = {"step": [], "step_n_toks": [], "mem": []}
        
        suffix = f"{which}_repeat_mem.json"
        
        for step in ckp_steps:
            step = step.replace("step", "step-")

            fn = os.path.join(datadir, ckp, f"pythia_{ckp}_{step}_{suffix}")
            with open(fn, 'r') as f:
                mem = json.load(f)
            
            out[ckp]["step"].append(step.replace("step-", "step "))
            out[ckp]["step_n_toks"].append((int(step.strip("step"))*N_TOKENS_PER_BATCH)/1e6)
            out[ckp]["mem"].append(mem["rs"])

        out[ckp]["step"] = np.array(out[ckp]["step"])
        out[ckp]["mem"] = np.array(out[ckp]["mem"])

    return out

# %%
def make_abstract_concrete_plot():

    # load data
    dct = load_abstract_concrete_data()

    dat = np.array(dct["rs"])  # (model, samples, token_position)
    dat = np.mean(dat[..., :], axis=-1)  # (model, samples)

    cond = np.array(dct["cond"])
    model = np.array(dct["model"])

    y_con = dat[cond == "Concrete", :]
    y_abs = dat[cond == "Abstract", :]

    # labels are the same across conditions
    model_labels = model[cond == "Concrete"]

    # (model, 2) array of means
    y_means = {
        "Concrete": trim_mean(y_con, proportiontocut=0.1, axis=1),
        "Abstract": trim_mean(y_abs, proportiontocut=0.1, axis=1)
    }

    def _trim_mean(x):
        return trim_mean(x, proportiontocut=0.1)

    # dict of 95% CI
    y_ci = {
        "Concrete": np.array([bootstrap((y_con[i, :],), _trim_mean,
                                        n_resamples=2000,
                                        confidence_level=0.95
                                        ).confidence_interval
                              for i in range(y_con.shape[0])]),
        "Abstract": np.array([bootstrap((y_abs[i, :],), _trim_mean,
                                        n_resamples=2000,
                                        confidence_level=0.95
                                        ).confidence_interval
                              for i in range(y_abs.shape[0])])
    }


    fig, ax = plt.subplots(figsize=(7.5, 5))

    width = 0.4
    multiplier = 0
    x = np.arange(len(y_means["Concrete"]))
    for cond in ("Concrete", "Abstract"):
        offset = width * multiplier

        y = y_means[cond]
        ci = y_ci[cond]

        yerr = np.array([y-ci[:, 0], ci[:, 1]-y])
        
        ax.bar(x + offset, y, yerr=yerr, 
               width=width, label=cond, alpha=0.8)
        
        multiplier += 1

    ax.set_ylim([70, 100])

    # plot legend outside of the plot
    ax.legend(title="Noun type", loc="upper left",
              frameon=False, fontsize=16, title_fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(visible=True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(model_labels, fontsize=14, rotation=25)
    ax.tick_params(axis="both", which="major", labelsize=16)

    ax.set_ylabel("Repeat loss change (%)", fontsize=18)
    ax.set_xlabel("Model size (Nr. of parameters)", fontsize=18)

    plt.tight_layout()  

    return fig

# %%
def make_timecourse_noun_type_plot(datadir:str):

    # load data
    dct1 = load_timecourse_data(datadir, "concrete")
    dct2 = load_timecourse_data(datadir, "abstract")
    
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True, sharey=True)

    data_dict = {
        "Concrete-14m": dct1["pythia-14m"],
        "Abstract-14m": dct2["pythia-14m"],
        "Concrete-31m": dct1["pythia-31m"],
        "Abstract-31m": dct2["pythia-31m"],
        "Concrete-70m": dct1["pythia-70m"],
        "Abstract-70m": dct2["pythia-70m"],
        "Concrete-160m": dct1["pythia-160m"],
        "Abstract-160m": dct2["pythia-160m"]
    }
    
    clrs = plt.cm.viridis_r(np.linspace(0, 1, 8))
    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

    data_dict = {
        "Concrete-14m": dct1["pythia-14m"],
        "Abstract-14m": dct2["pythia-14m"],
    }
    plot_lines(data_dict, clrs=clrs[0:2], marks=["-o", "-^"], ax=axes[0, 0])
    data_dict = {
        "Concrete-31m": dct1["pythia-31m"],
        "Abstract-31m": dct2["pythia-31m"],
    }

    # plot the difference as fill_between

    plot_lines(data_dict, clrs=clrs[0:2], marks=["-o", "-^"], ax=axes[0, 1])
    data_dict = {
        "Concrete-70m": dct1["pythia-70m"],
        "Abstract-70m": dct2["pythia-70m"],
    }
    plot_lines(data_dict, clrs=clrs[0:2], marks=["-o", "-^"], ax=axes[1, 0])

    data_dict = {
        "Concrete-160m": dct1["pythia-160m"],
        "Abstract-160m": dct2["pythia-160m"],
    }
    plot_lines(data_dict, clrs=clrs[0:2], marks=["-o", "-^"], ax=axes[1, 1])

    d = dct1["pythia-31m"]
    x = np.array([int(s.strip("step"))*N_TOKENS_PER_BATCH for s in d["step"]])+1
    x_total = int(d["step"][-1].strip("step"))*N_TOKENS_PER_BATCH

    fs = 14

    #ax.vlines(x_total, 0, 100, colors="black", linestyles="--", linewidth=1, alpha=0.8)
    #ax.text(x[-1], 0, f"{int(x_total/1e9)}B\n(100%)", fontsize=10, verticalalignment="bottom", horizontalalignment="right")

    # format human-friendly x-axis labels
    def _get_percent(x):
        return round(x / x_total * 100, 2)
    
    inset_ax1 = axes[1, 0].inset_axes(
        [0.5, 0.15, 0.5, 0.5],  # [x, y, width, height]
        xticklabels=[], yticklabels=[],
    )
    inset_ax2 = axes[1, 1].inset_axes(
        [0.5, 0.15, 0.5, 0.5],  # [x, y, width, height]
        xticklabels=[], yticklabels=[],
    )

    data_dict_inset1 = {
        "Concrete-70m": {"mem": dct1["pythia-70m"]["mem"], "step": dct1["pythia-70m"]["step"]},
        "Abstract-70m": {"mem": dct2["pythia-70m"]["mem"], "step": dct2["pythia-70m"]["step"]},
    }
    data_dict_inset2 = {
        "Concrete-160m": dct1["pythia-160m"],
        "Abstract-160m": dct2["pythia-160m"],
    }

    plot_lines(data_dict_inset1, clrs=clrs[0:2], marks=["-o", "-^"], ax=inset_ax1)
    plot_lines(data_dict_inset2, clrs=clrs[0:2], marks=["-o", "-^"], ax=inset_ax2)

    inset_ax1.set_xscale("log")
    inset_ax1.set_xlim([3e9, 25*1e9])
    inset_ax1.set_ylim([75, 95])
    inset_ax1.set_yticks([80, 90], labels=[80, 90], fontsize=9, color="black", alpha=0.7)
    inset_ax1.set_xticks([10e9, 20e9], labels=["10B", "20B"], fontsize=9, color="black", alpha=0.7)

    inset_ax2.set_xscale("log")
    inset_ax2.set_xlim([3e9, 25*1e9])
    inset_ax2.set_ylim([75, 95])
    inset_ax2.set_yticks([80, 90], labels=[80, 90], fontsize=9, color="black", alpha=0.7)
    inset_ax2.set_xticks([10e9, 20e9], labels=["10B", "20B"], fontsize=9, color="black", alpha=0.7)

    axes[1, 0].indicate_inset_zoom(inset_ax1, edgecolor="black", alpha=0.5, linestyle="--")
    axes[1, 1].indicate_inset_zoom(inset_ax2, edgecolor="black", alpha=0.5, linestyle="--")

    for a in (inset_ax1, inset_ax2):
        a.grid(visible=True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.8)
        a.tick_params(axis="both", which="major", labelsize=9)
        a.tick_params(axis="both", which="minor", labelbottom=False)
        for line in a.lines:
            line.set_markersize(6.5)

    for a in axes.flatten():
        a.set_xscale("log")
        a.set_xlim(400e6, x_total*1.1)
        xlabs = [f"{int(x/1e6)}M\n({_get_percent(x)}%)" if (x/1e9)<1 else f"{int(x/1e9)}B\n({_get_percent(x)}%)" for x in a.get_xticks()]
        a.set_xticklabels(xlabs, fontsize=10)


    for a in axes.flatten():
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(visible=True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.8)
        a.tick_params(axis="both", which="major", labelsize=10)
        for line in a.lines:
            line.set_markersize(7.5)

    fig.supxlabel("Training tokens elapsed", fontsize=fs)
    fig.supylabel("Repeat loss change (%)", fontsize=fs)

    axes[0, 0].set_title("14M", fontsize=fs)
    axes[0, 1].set_title("31M", fontsize=fs)
    axes[1, 0].set_title("70M", fontsize=fs)
    axes[1, 1].set_title("160M", fontsize=fs)

    h, _ = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(h, ["Concrete", "Abstract"],
              title="Noun type", 
              loc="best",
              frameon=False, fontsize=10, title_fontsize=10, ncol=1)

    plt.tight_layout()

    return fig

# %%

def plot_lines2(data_dict, clrs, marks, ax):

    i = 0
    # plot the performance
    for check_point, data in data_dict.items():
        
        x = np.array([int(s.strip("step"))*N_TOKENS_PER_BATCH for s in data["step"]])+1
        d = data["mem"]  # shape (training_step, checkpoints, list_position)

        y = np.mean(d[:, :], axis=-1)  # shape (training_step, timepoints)
        #y_mean = trim_mean(y, proportiontocut=0.1, axis=1)  # trim 20% extreme values
        
        #def _my_statistic(data):
        #    return trim_mean(data, proportiontocut=0.1)

        #time_steps = range(y.shape[0])

        #ci_all = [
        #    bootstrap(
        #        data=(y[i, :],), 
        #        statistic=_my_statistic,
        #        n_resamples=2000,
        #        confidence_level=0.95)
        #    for i in time_steps
        #    ]

        # plot ci error bars
        #ci = np.array([(ci.confidence_interval.low, ci.confidence_interval.high)
        #              for ci in ci_all]).T
        
        #ax.fill_between(x, ci[0], ci[1], color=clrs[i], alpha=0.25)

        # plot the data
        ax.plot(x, y, 
                f"{marks[i]}", color=clrs[i],
                markeredgecolor="white",
                markersize=10,
                linewidth=2.5,
                label=check_point.strip("pythia-").upper())
        
        i += 1

    return ax

# %%
def make_difference_timecourse(datadir:str):

    abstract = load_timecourse_data(datadir, "abstract")
    concrete = load_timecourse_data(datadir, "concrete")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    # set fontsize
    fs = 16

    # compute the difference between concrete and abstract
    data_dict = {}
    for k in abstract.keys():
        data_dict[k] = {
            "step": abstract[k]["step"],
            "mem": trim_mean(concrete[k]["mem"], proportiontocut=0.1, axis=1) - trim_mean(abstract[k]["mem"], proportiontocut=0.1, axis=1),
        }

    clrs = plt.cm.viridis_r(np.linspace(0, 1, len(data_dict)))
    marks = ["--o", "--s", "--v", "--^", "--D", "--P", "--X", "--d", "--p", "--*"]

    _ = plot_lines2(data_dict, clrs, marks, ax)

    d = list(data_dict.items())[0][1]
    x = np.array([int(s.strip("step"))*N_TOKENS_PER_BATCH for s in d["step"]])+1
    x_total = int(d["step"][-1].strip("step"))*N_TOKENS_PER_BATCH

    ax.set_xscale("log")
    ax.set_xlim(1e8, x_total*1.1)
    ax.set_xlabel("Training step (Nr. and % total training tokens elapsed)", fontsize=fs)
    ylab = "Concreteness advantage ($\Delta$%)\n($L^{r}_{concrete}-L^{r}_{abstract}$)"
    ax.set_ylabel(ylab, fontsize=fs)

    ax.legend(
        title="Model size", loc="upper left", frameon=False, 
        fontsize=12, title_fontsize=14, ncol=2)

    ax.vlines(x_total, 0, ax.get_ylim()[-1], colors="black", linestyles="--", linewidth=1, alpha=0.8)
    ax.text(x[-1], 0, f"{int(x_total/1e9)}B\n(100%)", fontsize=10, verticalalignment="bottom", horizontalalignment="right")

    # format human-friendly x-axis labels
    def _get_percent(x):
        return round(x / x_total * 100, 2)
    
    xlabs = [f"{int(x/1e6)}M\n({_get_percent(x)}%)" if (x/1e9)<1 else f"{int(x/1e9)}B\n({_get_percent(x)}%)" for x in ax.get_xticks()]
    ax.set_xticklabels(xlabs)

    ax.tick_params(axis="both", which="major", labelsize=fs)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(visible=True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.8)

    # add a vertical arrow along y-axis that says "better retrieval of concrete nouns"
    #ax.annotate(
    #    "", xy=(-0.11, 0.85), xytext=(-0.11, 0.15),
    #    xycoords="axes fraction", textcoords="axes fraction",
    #    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray")
    #)
    # add texte along the arrow
    #ax.text(-0.13, 0.5, "Better retrieval of concrete nouns",
    #        fontsize=12, verticalalignment="center", horizontalalignment="center", rotation=90, color="gray",
    #    transform=ax.transAxes
    #)

    plt.tight_layout()

    return fig


# %%
def make_dist_plot():

    dct = load_abstract_concrete_data()

    dat = np.array(dct["rs"])  # (model, samples, token_position)
    dat = np.mean(dat[..., :], axis=-1)  # (model, samples)

    cond = np.array(dct["cond"])
    model = np.array(dct["model"])

    model_sel = np.in1d(model, ["14M", "31M", "70M", "160M"])
    model = model[model_sel]
    cond = cond[model_sel]
    dat_sel = dat[model_sel, :]

    y_con = dat_sel[cond == "Concrete", :]
    y_abs = dat_sel[cond == "Abstract", :]

    df_con_dct = {m: y_con[i, :] for i, m in enumerate(model[cond == "Concrete"])}
    df_abs_dct = {m: y_abs[i, :] for i, m in enumerate(model[cond == "Abstract"])}

    df_con = pd.DataFrame(df_con_dct).melt(value_name="rs", var_name="model")
    df_con["type"] = "Concrete"
    df_abs = pd.DataFrame(df_abs_dct).melt(value_name="rs", var_name="model")
    df_abs["type"] = "Abstract"
    df = pd.concat([df_con, df_abs])

    # plot the distribution of memory traces using scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))

    sns.violinplot(
        data=df,
        x="model",
        y="rs",
        hue="type",
        split=True,
        gap=.2,
        inner="quart",
        density_norm="area",
        linewidth=1,
        cut=0,
        ax=ax
    )

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    def get_ranges(dat1, dat2):

        med = np.median(dat1)
        q25 = np.quantile(dat1, 0.25)
        q75 = np.quantile(dat1, 0.75)

        med_ = np.median(dat2)
        q25_ = np.quantile(dat2, 0.25)
        q75_ = np.quantile(dat2, 0.75)

        return (med, q25, q75), (med_, q25_, q75_)

    gap_half = 0.04
    for i in range(y_con.shape[0]):
        ranges1, ranges2 = get_ranges(y_con[i, :], y_abs[i, :])
        for d in zip(ranges1, ranges2):
            a, b = d
            ax.plot([i-gap_half, i+gap_half], [a, b], color="black", linestyle="--", linewidth=1, alpha=0.8)

    sns.move_legend(ax, "lower right", title="Noun type")

    sns.despine()
    ax.grid(visible=True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.8)

    ax.set_ylabel("Repeat loss change (%)", fontsize=14)
    ax.set_xlabel("Model size (Nr. of parameters)", fontsize=14)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)

    plt.tight_layout()

    return fig


# %%

def main(input_args:list=None):

    import argparse

    parser = argparse.ArgumentParser(description="Plot memory retrieval traces for Pythia models")
    parser.add_argument("--datadir", type=str, default=None, help="The folder containing the data files")
    parser.add_argument("--savedir", type=str, default=None, help="The folder to save the plot to")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.datadir = os.path.join(DATA_PATH.parent.parent, "data", "abst-conc") if args.datadir is None else args.datadir

    fig1 = make_abstract_concrete_plot()

    #fig2 = make_timecourse_noun_type_plot(args.datadir)

    fig2 = make_difference_timecourse(args.datadir)

    plt.show()

    #fig3 = make_dist_plot()

    if args.savedir is not None:
        
        fn = os.path.join(args.savedir, "fig_mem_abst-conc")
        logger.info(f"Saving figure to {fn} ...")
        save_png_pdf(fig1, fn)

        fn = os.path.join(args.savedir, "fig_mem_abst-conc_diff-timecourse")
        logger.info(f"Saving figure to {fn} ...")
        save_png_pdf(fig2, fn)

       # fn = os.path.join(args.savedir, "fig_mem_abst-conc_dist")
       # logger.info(f"Saving figure to {fn} ...")
       #save_png_pdf(fig3, fn)


# %%
if __name__ == "__main__":
    
    main()
