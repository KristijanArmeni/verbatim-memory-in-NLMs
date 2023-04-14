import os
import pandas as pd
from wm_suite.viz.func import make_point_plot, filter_and_aggregate
from wm_suite.viz.utils import load_csv_data, data_dir
import logging
import numpy as np
from matplotlib import pyplot as plt

# define dicts that contain text elements for plots
model_ids = {"gpt2": "a-10", "awd_lstm": "a-10", "w-12v2": "w-12v2"}
tags = {"gpt2": "gpt2", "awd_lstm": "awd_lstm", "w-12v2": "trf-w12v2"}
titles = {"gpt2": "Transformer (Radford et al, 2019)", "w-12v2": "Transformer (WT103, 12-layer)", "awd_lstm": "LSTM (Merity et al, 2017)"}
arcs = {"gpt2": "gpt2", "w-12v2": "gpt2", "awd_lstm": "awd_lstm"}
ylims={"gpt2": (None, None), "awd_lstm": (80, 115), "w-12v2": (60, 115)}
scenario = "sce1"

def generate_plot(model):

    logging.info(f"Generating plot for {model}")

    # ==== LOAD DATA ===== #

    if model == "gpt2":
        data = load_csv_data(model, datadir=data_dir, fname="output_gpt2_a-10_sce1.csv")
        data["model"] = "gpt2"
    
    elif model == "awd_lstm":
        data = load_csv_data(model=model, datadir=data_dir, fname="output_awd-lstm-3_a-10_sce1.csv")
        data["model"] = "awd_lstm"
        data.rename(columns={"word": "token"}, inplace=True)

    elif model == "w-12v2":
        data = load_csv_data(model=model, datadir=data_dir, fname=f"*{model}*sce1_1_*.csv").dropna()
        data["model"] = "gpt2"
        data.prompt_len = data.prompt_len.map({1: 8, 2: 30, 3:100, 4:200, 5:400})
        data.context = data.context.map({"sce1": "intact"})

    variables = [{"list_len": [3, 5, 7, 10]},
                 {"prompt_len": [8]},
                 {"context": ["intact"]},
                 {"marker_pos_rel": list(range(1, 10))}]

    # filter the dataframe and aggreage surprisal over lists (over the <marker_pos_rel> variable)
    data_, _ = filter_and_aggregate(datain=data, 
                                    model=data.model.unique()[0], 
                                    model_id=model_ids[model], 
                                    groups=variables, 
                                    aggregating_metric="mean")

    plot_size=(4, 3)
        
    grid, ax, stat = make_point_plot(data_frame=data_, estimator=np.median, x="list_len", y="x_perc", hue="condition", style="condition", 
                                     col="list", ylim=ylims[model],
                                     xlabel="Set size\n(n. tokens)", ylabel="Repeat surprisal\n(%)",
                                     suptitle=titles[model], scale=1, errwidth=1.5,
                                     legend=False, legend_out=True, custom_legend=True, legend_title="Second list",
                                     size_inches=plot_size)
    
    grid.fig.subplots_adjust(top=0.70)
    
    ax[0].set_title("Arbitrary list\n", fontsize=13)
    ax[1].set_title("Semantically coherent\nlist", fontsize=13)
    
    tick_fs, label_fs = 14, 14
    for a in ax:
        for label in (a.get_xticklabels() + a.get_yticklabels()): 
            label.set_fontsize(tick_fs)
        # if you want green xlabel, include this argument here: color='#23a952'
        a.set_xlabel(a.get_xlabel(), fontsize=label_fs)

    ax[0].set_ylabel("Repeat surprisal\n(%)", fontsize=label_fs)
    
    return grid, stat


def savefig(grid, savedir: str, figfname:str) -> None:

    logging.info("Saving {}".format(os.path.join(savedir, figfname)))
    grid.savefig(os.path.join(savedir, figfname + ".pdf"), transparent=True, bbox_inches="tight")
    grid.savefig(os.path.join(savedir, figfname + ".png"), dpi=300, bbox_inches="tight")


def savetable(stat, savedir: str, tablefname:str) -> None:

        # create a column with string formated and save the table as well
        stat = stat.round({"ci_min": 1, "ci_max": 1, "est": 1})
        strfunc = lambda x: str(x["est"]) + "% " + " " + "(" + str(x["ci_min"]) + "-" + str(x["ci_max"]) + ")"
        stat["report_str"] = stat.apply(strfunc, axis=1)

        # save the original .csv
        fname = os.path.join(savedir, tablefname)
        print("Writing {}".format(fname))
        stat.to_csv(fname)

        # save for latex
        stat.rename(columns={"hue": "Condition", "cond": "List", "xlabel": "Set-Size"}, inplace=True)
        stat = stat.pivot(index=["List", "Condition"], columns=["Set-Size"], values="report_str")
        stat.columns = stat.columns.astype(int)
        stat.sort_index(axis=1, ascending=True, inplace=True)
        tex = stat.to_latex(bold_rows=True,
                            label=f"tab:{tablefname}",
                            caption=f"Word list surprisal as a function of set size. We report the percentage of " + \
                            "list-averaged surprisal on second relative to first lists. Error bars denote 95\% confidence intervals around " \
                            "the group median (bootstrap estimate). " \
                            "The length of intervening text is fixed at 26 tokens.")

        # now save as .tex file
        fname = os.path.join(savedir, tablefname)
        print("Writing {}".format(fname))
        with open(fname, "w") as f:
            f.writelines(tex)

    
def main(input_args=None):

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    p, stat = generate_plot(model=args.model)

    figsavename = f"list-length_{scenario}_{tags[args.model]}"
    tablesavename = f"list-length_{scenario}_{tags[args.model]}_table.tex"

    savefig(p, savedir=args.savedir, figfname=figsavename)
    savetable(stat, savedir=args.savedir, tablefname=tablesavename)


if __name__ == "__main__":

    main()

