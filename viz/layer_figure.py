import os
import pandas as pd
from func import make_point_plot, filter_and_aggregate
from utils import load_csv_data, data_dir, savedir, table_savedir
import logging
import numpy as np
from matplotlib import pyplot as plt

# define dicts that contain text elements for plots
ylims = {"w-01v2": (50, None), "w-03v2": (50, None), "w-06v2": (50, None), "w-12v2": (50, None)}
titles = {"w-01v2": "1-layer", 
          "w-03v2": "3-layer", 
          "w-06v2": "6-layer",
          "w-12v2": "12-layer",
         }


def generate_plot(model_id):

    fn = f'*{model_id}_sce3*.csv'
    data = load_csv_data(model=model_id, datadir=os.path.join(data_dir, 'wt103_v2'), fname=fn)
    data["model"] = "gpt2"
    data.context = data.context.map({"sce3": "short"})
    data.prompt_len = data.prompt_len.map({1: 8})

    # ===== SELECT DATA ===== #
    variables = [{"list_len": [3, 5, 7, 10]},
                 {"prompt_len": [8]},
                 {"context": ["short"]},
                 {"marker_pos_rel": list(range(1, 10))}]

    
    data_, _ = filter_and_aggregate(datain=data.dropna(), model="gpt2", model_id=model_id, groups=variables, aggregating_metric="mean")

    plot_size=(5, 3)
    
    grid, ax, stat = make_point_plot(data_frame=data_, estimator=np.median, 
                                     x="list_len", y="x_perc", hue="condition", style="condition",
                                     col="list", ylim=ylims[model_id],
                                     xlabel="Set size\n(n. tokens)", ylabel="Repeat surprisal\n(%)",
                                     suptitle=titles[model_id], suptitle_fs=24, scale=1, errwidth=1.5,
                                     legend=False, legend_out=True, custom_legend=True, legend_title="Second list",
                                     size_inches=plot_size)
    
    grid.fig.subplots_adjust(top=0.70)
    ax[0].set_title("Arbitrary list\n")
    ax[1].set_title("Semantically coherent\nlist")
    
    plt.suptitle(titles[model_id], fontsize=24)

    # handle yticks and labels
    ax[0].set_yticks(list(range(ylims[model_id][0], 120, 20)))
    ax[1].set_yticks(list(range(ylims[model_id][0], 120, 20)))
    ax[0].set_yticklabels(list(range(ylims[model_id][0], 120, 20)))
        
    tick_fs, label_fs = 22, 22
    
    for a in ax:
        for label in (a.get_xticklabels() + a.get_yticklabels()): 
            label.set_fontsize(tick_fs)
        a.set_xlabel(a.get_xlabel(), fontsize=label_fs)
    
    ax[0].set_ylabel("Repeat surprisal\n(%)", fontsize=label_fs)

    return grid, stat


def savefig(grid, savedir, figfname):

    logging.info(f"Saving {figfname}")
    grid.savefig(os.path.join(savedir, figfname + ".pdf"), transparent=True, bbox_inches="tight")
    grid.savefig(os.path.join(savedir, figfname + ".png"), dpi=300, bbox_inches="tight")


def savetable(stat, savedir, tablefname):

    # create a column with string formated and save the table as well
    stat = stat.round({"ci_min": 1, "ci_max": 1, "est": 1})
    strfunc = lambda x: str(x["est"]) + "% " + "(" + str(x["ci_min"]) + "-" + str(x["ci_max"]) + ")"
    stat["report_str"] = stat.apply(strfunc, axis=1)

    # save the original .csv
    fname = os.path.join(savedir, tablefname + ".csv")
    print("Writing {}".format(fname))
    stat.to_csv(fname)

    # save for latex
    stat.rename(columns={"hue": "Condition", "cond": "List", "xlabel": "Set-Size"}, inplace=True)
    stat = stat.pivot(index=["List", "Condition"], columns=["Set-Size"], values="report_str")
    stat.columns = stat.columns.astype(int)
    stat.sort_index(axis=1, ascending=True, inplace=True)
    
    tex = stat.to_latex(bold_rows=True,
                        label=f"{tablefname}.tex",
                        caption="Word list surprisal as a function of set size. We report the percentage of " + \
                        "list-averaged surprisal on second relative to first lists. Ranges are 95\% confidence intervals around "\
                        "the observed median (bootstrap estimate).")

    # now save as .tex file
    fname = os.path.join(savedir, tablefname + ".tex")
    print("Writing {}".format(fname))
    with open(fname, "w") as f:
        f.writelines(tex)


def main(input_args=None):

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, choices=["w-01v2", "w-03v2", "w-06v2", "w-12v2"])
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    p, stat = generate_plot(model_id=args.model_id)

    figsavename = f"per-layer_sce3_trf-{args.model_id}"
    tablesavename = f"per-layer_sce3_trf-{args.model_id}_table"

    savefig(p, savedir=args.savedir, figfname=figsavename)
    savetable(stat, savedir=args.savedir, tablefname=tablesavename)

if __name__ == "__main__":

    main()