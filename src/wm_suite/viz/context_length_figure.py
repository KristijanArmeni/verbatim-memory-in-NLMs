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
        data = load_csv_data(model=model, datadir=data_dir, fname=f"*{model}*sce1_*10*.csv").dropna()
        data["model"] = "gpt2"
        data.prompt_len = data.prompt_len.map({1: 8, 2: 30, 3:100, 4:200, 5:400})
        data.context = data.context.map({"sce1": "intact"})

    # ===== SELECT AND AVERAGE DATA ===== #
    variables = [{"prompt_len": [8, 100, 200, 400]},
                {"list_len": [10]},
                {"context": ["intact"]},
                {"marker_pos_rel": list(range(1, 10))}]

    data_, _ = filter_and_aggregate(datain=data, 
                                    model=data.model.unique()[0], 
                                    model_id=model_ids[model], 
                                    groups=variables, 
                                    aggregating_metric="mean")

    data_.prompt_len = data_.prompt_len.astype(int)
    # rename prompt length values to more meaningful ones
    prompt_len_map = {8: 26, 30: 47, 100: 99, 200: 194, 400: 435}
    data_.prompt_len = data_.prompt_len.map(prompt_len_map)

    plot_size=(4, 3)
    
    grid, ax, stat = make_point_plot(data_frame=data_, estimator=np.median, x="prompt_len", y="x_perc", hue="condition", style="condition",
                                     col="list", ylim=ylims[model],
                                     xlabel="Intervening text\nlen. (n. tokens)", ylabel="Repeat surprisal\n(%)",
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
        # if you need to color the text label, provide this input argument here: color="#FF8000", 
        a.set_xlabel(a.get_xlabel(), fontsize=label_fs)

    ax[0].set_ylabel("Repeat surprisal\n(%)", fontsize=label_fs)

    return grid, stat


def savefig(grid, savedir, figfname):

        fn = os.path.join(savedir, figfname)
        print(f"Saving {fn}")
        grid.savefig(fn + ".pdf", transparent=True, bbox_inches="tight")
        grid.savefig(fn + ".png", dpi=300, bbox_inches="tight")
    

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
        stat.rename(columns={"hue": "Condition", "cond": "List", "xlabel": "Intervening text len."}, inplace=True)
        stat = stat.pivot(index=["List", "Condition"], columns=["Intervening text len."], values="report_str")
        stat.columns = stat.columns.astype(int)
        stat.sort_index(axis=1, ascending=True, inplace=True)
        
        tex = stat.to_latex(bold_rows=True,
                            label=f"tab:{tablefname}",
                            caption="Word list surprisal as a function of intervening text size. We report the percentage of " + \
                            "list-averaged surprisal on second relative to first lists. Error bars denote 95\% confidence intervals around "\
                            "the group median (bootstrap estimate). "\
                            "The list length is fixed at 10 tokens.")

        # now save as .tex file
        fname = os.path.join(savedir, f"{tablefname}" + ".tex")
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

    figsavename = f"context-length_{scenario}_{tags[args.model]}"
    tablesavename = f"context-length_{scenario}_{tags[args.model]}_table"

    savefig(p, savedir=args.savedir, figfname=figsavename)
    savetable(stat, savedir=args.savedir, tablefname=tablesavename)


if __name__ == "__main__":

    main()
