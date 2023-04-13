import os
import pandas as pd
from func import make_point_plot, filter_and_aggregate
from utils import load_csv_data, data_dir
import logging
import numpy as np
from matplotlib import pyplot as plt

# define dicts that contain text elements for plots
model_ids = {"gpt2": "a-10", "w-12v2": "w-12v2"}
suptitles = {"punctuation": "Comma as cue", 
             "name": "'John' instead of 'Mary'", 
              "shuffled-preface": "Shuffled preface string",
              "shuffled-prompt": "Shuffled prompt string"}
savetags = {"w-12v2": "trf-w12v2", "gpt2": "trf-a10"}
ylims = {"w-12v2": (55, 115), "gpt2": (None, None)}
label2vignette = {"punctuation": "sce5", "name": "sce6", "shuffled-preface": "sce4", "shuffled-prompt": "sce7"}

def generate_plot(model, experiment):

    vignette = label2vignette[experiment]

    logging.info(f"Generating plot for {model}")

    # ==== LOAD DATA ===== #

    if model == "gpt2":
        data = load_csv_data(model, datadir=data_dir, fname=f"output_gpt2_a-10_{vignette}.csv")
        data["model"] = "gpt2"
        data["context"] = experiment

    elif model == "w-12v2":
        data = load_csv_data(model=model, datadir=data_dir, fname=f"*_w-12v2_{vignette}*.csv").dropna()
        data["model"] = "gpt2"
        data.prompt_len = data.prompt_len.map({1: 8, 2: 30, 3:100, 4:200, 5:400})
        data["context"] = experiment

    variables = [{"list_len": [3, 5, 7, 10]},
                {"prompt_len": [8]},
                {"context": [experiment]},
                {"marker_pos_rel": list(range(1, 10))}]

    data_, _ = filter_and_aggregate(datain=data, 
                                    model=data.model.unique()[0], 
                                    model_id=model_ids[model], 
                                    groups=variables, 
                                    aggregating_metric="mean")

    plot_size=(4, 3)
    
    grid, ax, stat = make_point_plot(data_frame=data_, x="list_len", y="x_perc", hue="condition", style="condition", col="list", ylim=ylims[model],
                                     estimator=np.median,
                                     xlabel="Set size\n(n. tokens)", ylabel="Repeat surprisal\n(%)",
                                     suptitle=suptitles[experiment], suptitle_fs=24, scale=1, errwidth=1.5,
                                     legend=False, legend_out=True, custom_legend=True, legend_title="Second list",
                                     size_inches=plot_size)
    
    grid.fig.subplots_adjust(top=0.70)
    
    ax[0].set_title("Arbitrary list\n")
    ax[1].set_title("Semantically coherent\nlist")
    
    for a in ax:
        for label in (a.get_xticklabels() + a.get_yticklabels()): 
            label.set_fontsize(22)
        a.set_xlabel(a.get_xlabel(), fontsize=22)

    ax[0].set_ylabel("Repeat surprisal\n(%)", fontsize=22)

    return grid, stat


def savefig(grid, savedir, figfname):

    logging.info(f"Saving {figfname}")
    grid.savefig(os.path.join(savedir, figfname + ".pdf"), transparent=True, bbox_inches="tight")
    grid.savefig(os.path.join(savedir, figfname + ".png"), dpi=300, bbox_inches="tight")


def savetable(stat, savedir, tablefname):

    # create a column with string formated and save the table as well
    stat = stat.round({"ci_min": 1, "ci_max": 1, "est": 1})
    strfunc = lambda x: str(x["est"]) + "% " + " " + "(" + str(x["ci_min"]) + "-" + str(x["ci_max"]) + ")"
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
                        label=f"tab:{tablefname}",
                        caption="Word list surprisal as a function of set size when ':' token is replaced by ',' in preface in prompt strings " + \
                        "We report the percentage of list-averaged surprisal on second relative to first lists. Ranges are 95\% confidence intervals around " \
                        "the observed median (bootstrap estimate). " \
                        "The length of intervening text is fixed at 26 tokens.")

    # now save as .tex file
    fname = os.path.join(savedir, tablefname + ".tex")
    print("Writing {}".format(fname))
    with open(fname, "w") as f:
        f.writelines(tex)


def main(input_args=None):

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt2", "w-12v2"])
    parser.add_argument("--experiment", type=str, choices=["punctuation", "name", "shuffled-preface", "shuffled-prompt"])
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    p, stat = generate_plot(model=args.model, experiment=args.experiment)

    figsavename = f"{args.experiment}_trf-{model_ids[args.model]}"
    tablesavename = f"{args.experiment}_trf-{model_ids[args.model]}_table"

    savefig(p, savedir=args.savedir, figfname=figsavename)
    savetable(stat, savedir=args.savedir, tablefname=tablesavename)

if __name__ == "__main__":

    main()