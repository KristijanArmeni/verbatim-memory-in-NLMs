from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .corr import fetch_data, min_max
from .fig_mem import N_TOKENS_PER_BATCH
from ..utils import save_png_pdf


CORR_DF = Path("c://users/karmeni1/project/lm-mem/data/pythia/learning_traj_corr.tsv")

"""
def get_color_df(d1: pd.DataFrame) -> pd.DataFrame:

    # final rho for all models tasks
    final_step = TIMESTEPS[-1]
    final_rho = d1.loc[d1.step == final_step].reset_index()

    # map viridis onto rhos
    
    # normalize


    out = {f"{row[1].model}-{row[1].benchmark}": c for row, c in zip(final_rho.iterrows(), colors)}

    for i in range(colors.shape[-1]):
        final_rho[f"c{i}"] = colors[:, i]

    return cmap, norm, final_rho
""" 

reformat_benchmarks = {
    "arc_challenge": "ARC (challenge)",
    "arc_easy": "ARC (easy)",
    "lambada_openai": "Lambada (OpenAI)",
    "logiqa": "LogiQA",
    "mmlu-STEM": "MMLU (STEM)",
    "mmlu-humanities": "MMLU (Humanities)",
    "mmlu-other (business, health, misc.)": "MMLU (Other)",
    "mmlu-social sciences": "MMLU (Soc. sci.)",
    "piqa": "PiQA",
    "sciq": "SciQ",
    "winogrande": "Winogrande",
    "wsc": "WSC"
}

chances = {
    "ARC (challenge)": 0.25,
    "ARC (easy)": 0.25,
    "Lambada (OpenAI)": 0.016,  # random guess, see Table 1: https://arxiv.org/pdf/1606.06031
    "MMLU (STEM)": 0.25,
    "MMLU (Humanities)": 0.25,
    "MMLU (Other)": 0.25,
    "MMLU (Soc. sci.)": 0.25,
    "LogiQA": 0.25,
    "PiQA": 0.5,
    "SciQ": 0.25,
    "Winogrande": 0.5,
    "WSC": 0.5
}

def plot_corr(corr_df, models, ax=None, figsize=(3, 4)):

    labelfs = 18

    cmap = plt.cm.Blues
    vmin, vmax = 0, corr_df.final_acc.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # get colors rgb (n_models*n_tasks, 4)
    clrs = cmap(norm(corr_df.final_acc.to_numpy()))  # n_models*n_tasks, 4

    # add color codes columns to corr df
    for i in range(clrs.shape[-1]):
        corr_df[f"c{i}"] = clrs[:, i]

    xmin, xmax = corr_df.rho.min(), corr_df.rho.max()
    xmin, xmax = -1, 1

    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        
        model = models[i]

        sel_df = corr_df[corr_df.model == model].copy()

        # sort benchmarks alphabetically
        sel_df.benchmark = sel_df.benchmark.map(reformat_benchmarks)
        sel_df = sel_df.sort_values(by="benchmark", ascending=False)

        # color-code barplots based on final correlations score
        for i, row in sel_df.iterrows():
            
            hatch = "///" if row.final_acc <= chances[row.benchmark] else None
            ax.barh(row.benchmark, row.rho, color=row.loc[["c0", "c1", "c2", "c3"]], hatch=hatch)

            # plot CI errorbars
            xerrs = ([row.rho - row.CI_low], [row.CI_high - row.rho])
            ax.errorbar(row.rho, row.benchmark, xerr=xerrs, color='gray')

        ax.set_xticks([-1, 0, 1])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # colorbar
        ax.set_title(f"{model.upper()}", fontsize=labelfs)
        ax.set_xlim(xmin-0.05, xmax+0.05)

        ax.grid(axis='x', which="both", linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label='Benchmark accuracy\nbelow chance/baseline')
    fig.legend(loc="outside lower right", handles=[patch], fontsize=14, frameon=False)

    for ax in axes[1::]:
        ax.spines['left'].set_visible(False)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=labelfs)

    axes[0].set_ylabel("Benchmark task", fontsize=labelfs)

    xlab = "Spearman correlation coefficient"
    if len(models) == 1:
        axes[0].set_xlabel(xlab, fontsize=labelfs)
        cbar_ax = axes[0]
    else:
        fig.supxlabel(xlab, fontsize=labelfs)
        divider = make_axes_locatable(axes[-1])
        cbar_ax = divider.append_axes("right", size="10%", pad="15%")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_ylabel("Benchmark accuracy\n(fully-trained)", fontsize=labelfs)
    cbar.ax.tick_params(labelsize=labelfs)

    return fig, axes


def plot_corr_data(bench_df, mem_df, corr_df, model, tasks, minmax=False, figsize=(10, 4.5)):

    # find top-5 correlations and bottom 5
    n_ckp = 27
    n_tasks = len(tasks)

    markers = ["s", "v", "D", "P", "X"]
    assert n_tasks <= len(markers), "Too many tasks for markers"
    
    datay = np.zeros((n_ckp, n_tasks))
    datay2 = np.zeros((n_ckp, 1))
    datax = np.zeros((n_ckp, n_tasks))
    datax2 = np.zeros((n_ckp, 1))

    selcorr = (corr_df.model == model) & (corr_df.benchmark.isin(tasks))
    corr_vals = corr_df.loc[selcorr, ["rho", "benchmark"]]

    for i, tsk in enumerate(tasks):

        selrows = (bench_df.model == model) & (bench_df.category == tsk)
        assert selrows.sum() == n_ckp, f"Length mismatch for {model} and {tsk} ({selrows.sum()} vs {n_ckp})"
        if minmax:
            datay[:, i] = min_max(bench_df.loc[selrows].acc)            
        else:
            datay[:, i] = bench_df.loc[selrows].acc * 100

        datax[:, i] = bench_df.loc[selrows].step * N_TOKENS_PER_BATCH
    
    if minmax:
        datay2[:, 0] = min_max(mem_df.loc[mem_df.model == model].mem)
    else:
        datay2[:, 0] = mem_df.loc[mem_df.model == model].mem
    datax2[:, 0] = mem_df.loc[mem_df.model == model].step * N_TOKENS_PER_BATCH

    # plot time courses
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    vals = {t: corr_vals.loc[corr_vals.benchmark == t].rho.item() for t in tasks}

    ylabs = [f"{reformat_benchmarks[t]} [rho = {vals[t]:.2f}]" for t in tasks]

    ax.plot(datax2, datay2, '--', marker='o', label="Verbatim retrieval", color="black")

    for i, tsk in enumerate(tasks):
        ax.plot(datax[:, i], datay[:, i], marker=markers[i], linestyle="-", label=ylabs[i])
    
    ax.set_xscale("log")

    x_total = datax[-1, -1]
    def _get_percent(x):
        return round(x / x_total * 100, 2)

    xlabs = [f"{int(x/1e6)}M\n({_get_percent(x)}%)" if (x/1e9)<1 else f"{int(x/1e9)}B\n({_get_percent(x)}%)" for x in ax.get_xticks()]
    #ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(xlabs)

    ax.grid(visible=True)
    ax.spines['top'].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fs = 16

    ax.legend(loc="upper left", fontsize=12)

    ax.set_title(f"Learning trajectories ({model.upper()} model)", fontsize=fs)
    ax.set_ylabel("Performance\n(min-max normalized)", fontsize=fs)
    ax.set_xlabel("Training step (Nr. and % train tokens elapsed)", fontsize=fs)

    ax.tick_params(axis='both', which='major', labelsize=fs)

    return fig, ax


def main():

    #savedir = Path("c://users/karmeni1/project/lm-mem/fig/pythia")
    savedir = None

    corr_df = pd.read_csv(CORR_DF, sep="\t")
    
    fig, axes = plot_corr(
        corr_df,
        models=["160m", "410m", "1.4b", "2.8b", "6.9b", "12b"],
        figsize=(15, 5.5)
    )
    
    plt.tight_layout()

    if savedir is not None:
        save_png_pdf(fig, str(Path(savedir, "pythia_benchmark_corr")))
    
    bench_df, mem_df = fetch_data()
    tasks = ["arc_easy", "lambada_openai", "sciq", "winogrande"]

    fig, ax = plot_corr_data(bench_df, mem_df, corr_df, "12b", tasks, minmax=True, figsize=(7, 4.5))
    plt.tight_layout()
    plt.show()

    if savedir is not None:
        save_png_pdf(fig, str(Path(savedir, "pythia_12b_learning_traj")))
    

if __name__ == "__main__":

    main()