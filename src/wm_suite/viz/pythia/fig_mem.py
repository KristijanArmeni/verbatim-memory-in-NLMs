#%%
import os
import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import trim_mean, bootstrap
from wm_suite.paths import DATA_PATH
from wm_suite.viz.utils import save_png_pdf
from tqdm import tqdm

# %%
steps = [
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

ckp_steps_dict = {
    "pythia_14m":  steps,
    "pythia_31m": steps,
    "pythia_70m": steps,
    "pythia_160m": steps,
    "pythia_410m": steps,
    "pythia_1b": steps,
    "pythia_6.9b": steps,
    "pythia_12b": steps,
}

N_TOKENS_PER_BATCH = 2097152  # as per the Pythia paper

#%%
def _load_data(datadir, which="repeat"):

    out = {}

    for ckp, ckp_steps in ckp_steps_dict.items():

        out[ckp] = {"step": [], "step_n_toks": [], "mem": []}
        suffix = "repeat_mem.json"

        if which == "control":
            suffix = "control_mem.json"
        
        for step in tqdm(ckp_steps, desc=f"Loading steps for {ckp}"):
            fn = os.path.join(datadir, ckp, f"{ckp.replace('_', '-')}_{step}_{suffix}")
            with open(fn, 'r') as f:
                mem = json.load(f)
            
            out[ckp]["step"].append(step)
            out[ckp]["step_n_toks"].append((int(step.strip("step"))*N_TOKENS_PER_BATCH)/1e6)
            out[ckp]["mem"].append(mem["rs"])

        out[ckp]["step"] = np.array(out[ckp]["step"])
        out[ckp]["mem"] = np.array(out[ckp]["mem"])

    return out


def plot_lines(data_dict, clrs, marks, ax):

    i = 0
    # plot the performance
    for check_point, data in data_dict.items():
        
        x = np.array([int(s.strip("step"))*N_TOKENS_PER_BATCH for s in data["step"]])+1
        d = data["mem"]  # shape (training_step, samples, list_position)

        y = np.mean(d[:, :, :], axis=-1)
        y_mean = trim_mean(y, proportiontocut=0.1, axis=1)  # trim 20% extreme values
        
        def _my_statistic(data):
            return trim_mean(data, proportiontocut=0.1)

        time_steps = range(y.shape[0])

        ci_all = [
            bootstrap(
                data=(y[i, :],), 
                statistic=_my_statistic,
                n_resamples=2000,
                confidence_level=0.95)
            for i in time_steps
            ]

        # plot ci error bars
        ci = np.array([(ci.confidence_interval.low, ci.confidence_interval.high)
                      for ci in ci_all]).T
        
        ax.fill_between(x, ci[0], ci[1], color=clrs[i], alpha=0.25)

        # plot the data
        ax.plot(x, y_mean, 
                f"{marks[i]}", color=clrs[i],
                markeredgecolor="white",
                markersize=9,
                label=check_point.strip("pythia_").upper())
        
        i += 1
    
    return ax


# %%
def plot_learning_curve(data_dict, cmap:str="viridis") -> tuple:

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # set fontsize
    fs = 14

    if cmap == "viridis":
        clrs = plt.cm.viridis_r(np.linspace(0, 1, len(data_dict)))
    elif cmap == "tab10":
        clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    marks = ["--o", "--s", "--v", "--^", "--D", "--P", "--X", "--d"]

    ax = plot_lines(data_dict, clrs, marks, ax)

    d = list(data_dict.items())[0][1]
    x = np.array([int(s.strip("step"))*N_TOKENS_PER_BATCH for s in d["step"]])+1
    x_total = int(d["step"][-1].strip("step"))*N_TOKENS_PER_BATCH

    ax.set_xscale("log")
    ax.set_xlim(1e8, x_total*1.1)
    ax.set_xlabel("Training step (Nr. and % total training tokens elapsed)", fontsize=fs)
    ylab = "Repeat loss change (%)"
    ax.set_ylabel(ylab, fontsize=fs)

    #ax.set_title("Transformer noun retrieval across time and scale",
    #                 fontsize=fs)

    ax.vlines(x_total, 0, 100, colors="black", linestyles="--", linewidth=1, alpha=0.8)
    ax.text(x[-1], 0, f"{int(x_total/1e9)}B\n(100%)", fontsize=10, verticalalignment="bottom", horizontalalignment="right")

    # format human-friendly x-axis labels
    def _get_percent(x):
        return round(x / x_total * 100, 2)
    
    xlabs = [f"{int(x/1e6)}M\n({_get_percent(x)}%)" if (x/1e9)<1 else f"{int(x/1e9)}B\n({_get_percent(x)}%)" for x in ax.get_xticks()]
    ax.set_xticklabels(xlabs)

    ax.grid(visible=True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(title="Model size", loc="upper left", frameon=False, fontsize=12, title_fontsize=12, ncol=2)

    ax.tick_params(axis="both", which="major", labelsize=fs)

    return fig, ax

# %%
def make_plot(datadir:str, condition: str="repeat") -> tuple:

    out = _load_data(datadir, which=condition)

    fig1, ax = plot_learning_curve(data_dict=out)

    plt.tight_layout()

    fig2, ax = plot_per_word_position(datadict=out)

    plt.tight_layout()

    return fig1, fig2

# %%
def plot_curve_per_list_position(datadir:str, condition: str="repeat") -> tuple:

    out = _load_data(datadir, which=condition)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    # fondsizes etc.
    fs = 14

    clrs = plt.cm.viridis_r(np.linspace(0, 1, len(out)))
    marks = ["o", "s", "v", "^", "D", "P", "X", "d"] #"p", "h", "8", "H", "*", ">", "<", "1", "2", "
    
    i = 0
    # plot the performance
    for check_point, data in out.items():
        
        x = np.array([int(s.strip("step"))*N_TOKENS_PER_BATCH for s in data["step"]])+1
        d = data["mem"]  # shape (training_step, samples, list_position)

        for idx, k in enumerate(range(d.shape[-1])):

            y = np.median(d[:, :, k], axis=1)  # median across observations
            
            # plot the data
            ax[idx].plot(x, y, f"--{marks[i]}", color=clrs[i], label=check_point.strip("pythia_").upper())
        
        i += 1

    ax[0].set_title("First token", fontsize=fs)
    ax[1].set_title("Second token", fontsize=fs)
    ax[2].set_title("Third token", fontsize=fs)


    x_total = int(steps[-1].strip("step"))*N_TOKENS_PER_BATCH

    for a in ax:
        a.set_xscale("log")
        a.set_xlim(1e7, x_total*1.1)
    
    fig.supxlabel("Training step\n(Nr. and % tokens elapsed since start)", fontsize=fs)
    ylab = "Memory retrieval trace (%)\n$1 - \\frac{\mathrm{loss}(\mathrm{list}_{2})}{\mathrm{loss}(\mathrm{list}_{1})}$"
    ax[0].set_ylabel(ylab, fontsize=fs)

    if condition == "control":
        fig.suptitle("Verbatim retrieval across time and scale\n(control condition -- no noun repetition)",
                     fontsize=fs)
    else:
        fig.suptitle("Verbatim retrieval across time and ascale",
                     fontsize=fs)

    for a in ax:
        a.vlines(x_total, 0, 100, colors="black", linestyles="--", linewidth=1, alpha=0.8)
        a.text(x[-1], 0, f"{int(x_total/1e9)}B\n(100%)", fontsize=10, verticalalignment="bottom", horizontalalignment="right")

    # format human-friendly x-axis labels
    def get_percent(x):
        return round(x / x_total * 100, 2) if x / x_total * 100 < 1 else int(round(x / x_total * 100, 0))
    xlabs = [f"{int(x/1e6)}M\n({get_percent(x)}%)" if (x/1e9)<1 else f"{int(x/1e9)}B\n({get_percent(x)}%)" for x in ax[0].get_xticks()]
    xlabs[1] = "10M\n(< 0.03%)"

    for a in ax:
        a.set_xticklabels(xlabs)
        a.grid(visible=True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.8)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    ax[0].legend(title="Model size", loc="upper left", frameon=False, fontsize=11, title_fontsize=11, ncol=2)

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()

    return fig, ax

# %%

def plot_per_word_position(datadict) -> tuple:

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 4.5), sharex=True, sharey=True)

    # fondsizes etc.
    fs = 14

    clrs = plt.cm.viridis_r(np.linspace(0, 1, len(datadict)))
    marks = ["o", "s", "v", "^", "D", "P", "X", "d"] #"p", "h", "8", "H", "*", ">", "<", "1", "2", "
        
    i = 0
    # plot the performance
    for check_point, data in datadict.items():
        
        y = data["mem"][-1, :, :]  # pick last training step, shape = (observations, list_position)
        y_mean = trim_mean(y, proportiontocut=0.1, axis=0)  # trim 20% extreme values
        
        def _my_statistic(data):
            return trim_mean(data, proportiontocut=0.1)

        x = range(y.shape[1])  # (list_position,)

        ci_all = [
            bootstrap(
                data=(y[:, j],), 
                statistic=_my_statistic,
                n_resamples=2000,
                confidence_level=0.95)
            for j in x
        ]

        # plot ci error bars
        ci = np.array([(ci.confidence_interval.low, ci.confidence_interval.high)
                      for ci in ci_all]).T
        
        # plot error bars
        ax.errorbar(x, y_mean, yerr=[y_mean-ci[0], ci[1]-y_mean],
                    fmt=f"--{marks[i]}", color=clrs[i],
                    markeredgecolor="white",
                    markersize=9,
                    label=check_point.strip("pythia_").upper())
        
        i += 1

    ax.set_xlabel("Word position in repeated list", fontsize=fs)
    ax.set_ylabel("Repeat loss change (%)", fontsize=fs)

    # add an arrow here
    
    ax.set_ylim([50, 100])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}\n" for i in range(len(x))], fontsize=fs)

    ax.grid(visible=True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(title="Model size", loc="best", frameon=False, fontsize=12, title_fontsize=12, ncol=2)

    ax.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()

    return fig, ax

# %%
# convert data to csv
def data_to_csv(output:dict) -> pd.DataFrame:
    dfs = []
    for ckp, data in output.items():
        dftmp = pd.DataFrame(data)
        dftmp["size"] = ckp.strip("pythia_").upper()
        dftmp["mem"] = 100 - dftmp["mem"]
        dfs.append(dftmp)

    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={"mem": "memory"}, inplace=True)

    return df


# %%
def main(input_args:list=None):

    import argparse

    parser = argparse.ArgumentParser(description="Plot memory retrieval traces for Pythia models")
    parser.add_argument("--condition", type=str, default="repeat", help="Condition to plot (repeat or control)")
    parser.add_argument("--datadir", type=str, default=None, help="The folder containing the data files")
    parser.add_argument("--savedir", type=str, default=None, help="The folder to save the plot to")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args(input_args)

    args.datadir = os.path.join(DATA_PATH.parent.parent, "data", "pythia") if args.datadir is None else args.datadir

    fig1, fig2 = make_plot(datadir=args.datadir, condition=args.condition)

    plt.show()

    #fig2, ax2 = plot_curve_per_list_position(datadir=args.datadir, condition=args.condition)

    #plt.show()

    #fig3, ax3 = plot_per_word_position(datadir=args.datadir, condition=args.condition)

    #plt.show()

    if args.savedir is not None:

        fn = os.path.join(args.savedir, f"fig_mem_{args.condition}")
        save_png_pdf(fig1, fn)

        fn = os.path.join(args.savedir, f"fig_mem_{args.condition}_per_position")
        save_png_pdf(fig2, fn)

        #fn2 = os.path.join(args.savedir, f"fig_mem_per_timestep_{args.condition}")
        #save_png_pdf(fig2, fn2)

        #fn = os.path.join(args.savedir, f"pythia_mem_{args.condition}.csv")
        #df.to_csv(fn, index=False, sep="\t")



if __name__ == "__main__":

    main()