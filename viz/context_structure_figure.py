import os
import pandas as pd
from func import make_bar_plot, filter_and_aggregate
from utils import load_csv_data, data_dir, savedir, table_savedir
import logging
import numpy as np
from matplotlib import pyplot as plt

# define dicts that contain text elements for plots
model_ids = {"gpt2": "a-10", "awd_lstm": "a-10", "w-12v2": "w-12v2"}
tags = {"gpt2": "gpt2", "awd_lstm": "awd_lstm", "w-12v2": "trf-w12v2"}
titles = {"gpt2": "Transformer (Radford et al, 2019)", "w-12v2": "Transformer (WT103, 12-layer)", "awd_lstm": "LSTM (Merity et al, 2017)"}
arcs = {"gpt2": "gpt2", "w-12v2": "gpt2", "awd_lstm": "awd_lstm"}
ylims={"gpt2": (None, None), "awd_lstm": (60, None), "w-12v2": (60, None)}

def generate_plot(model):

    tmp = []

    for sce in ["sce1", "sce2", "sce1rnd"]:
        
        if model == "awd_lstm":
            logging.info(f"Loading output_awd-lstm-3_a-10_{sce}.csv")
            dat_ = load_csv_data(model=model, datadir=data_dir, fname=f"output_awd-lstm-3_a-10_{sce}.csv")
            dat_["model"] = model
            dat_.rename(columns={"word": "token"}, inplace=True)
            tmp.append(dat_)

        elif model == "gpt2":
        
            logging.info(f"Loading output_gpt2_a-10_{sce}.csv")
            dat_ = load_csv_data(model=model, datadir=data_dir, fname=f"output_gpt2_a-10_{sce}.csv")
            dat_["model"] = model
            tmp.append(dat_)
        
        elif model == "w-12v2":
            fn3 = f'*trf_w-12v2_{sce}_5_10*.csv'
            logging.info(f"Loading matches to {fn3}")
            dat_ = load_csv_data(model=model, 
                       datadir=os.path.join(data_dir, "wt103_v2"), 
                       fname=fn3).dropna()
            dat_["model"] = "gpt2"
            dat_.prompt_len = dat_.prompt_len.map({5:400})
            dat_.context = dat_.context.map({"sce1": "intact", "sce1rnd": "scrambled", "sce2": "incongruent"})
            tmp.append(dat_)

    data = pd.concat(tmp)

    # ===== SELECT DATA ===== #
    variables = [{"context": ["intact", 'scrambled', 'incongruent']},
                {"prompt_len": [400]},
                {"list_len": [10]},
                {"marker_pos_rel": list(range(1, 10))}]

    data_, _ = filter_and_aggregate(datain=data, 
                                    model=data.model.unique()[0], 
                                    model_id=model_ids[model], 
                                    groups=variables, 
                                    aggregating_metric="mean")

    plot_size=(6, 3)
    
    grid, ax, stat = make_bar_plot(data_frame=data_, estimator=np.median, x="context", y="x_perc", hue="condition", col="list", ylim=ylims[model],
                                   xlabel="Intervening text", ylabel="Repeat surprisal\n(%)",
                                   suptitle=titles[model],
                                   legend=False, legend_out=True, legend_title="Second list",
                                   size_inches=plot_size)
    
    grid.fig.subplots_adjust(top=0.70)
    ax[0].set_title("Arbitrary list\n", fontsize=16)
    ax[1].set_title("Semantically coherent\nlist", fontsize=16)
    
    xlabels_capitalized = [text.get_text().capitalize() for text in ax[0].get_xticklabels()]
    ax[0].set_xticklabels(labels=xlabels_capitalized, rotation=20)
    ax[1].set_xticklabels(labels=xlabels_capitalized, rotation=20)

    for a in ax:
        a.tick_params(labelsize=16)

    return grid, stat


def savefig(grid, savedir, figfname):

    fn = os.path.join(savedir, figfname)
    print(f"Saving {fn}")
    grid.savefig(os.path.join(savedir, fn + ".pdf"), transparent=True, bbox_inches="tight")
    grid.savefig(os.path.join(savedir, fn + ".png"), dpi=300, bbox_inches="tight")


def savetable(stat, savedir, tablefname):
    
    # create a column with string formated and save the table as well
    stat = stat.round({"ci_min": 1, "ci_max": 1, "median": 1})
    strfunc = lambda x: str(x["median"]) + " " + "(" + str(x["ci_min"]) + "-" + str(x["ci_max"]) + ")"
    stat["report_str"] = stat.apply(strfunc, axis=1)

    # save the original .csv
    fname = os.path.join(savedir, tablefname + ".csv")
    print("Writing {}".format(fname))
    stat.to_csv(fname)

    # save for latex
    stat.list = stat.list.str.capitalize()
    stat.rename(columns={"condition": "Condition", "list": "List", "context": "Context"}, inplace=True)
    stat = stat.pivot(index=["List", "Condition"], columns=["Context"], values="report_str")
    tex = stat.to_latex(bold_rows=True,
                        label=f"tab:{tablefname}",
                        caption="Word list surprisal as a function of intervening context. We report the percentage of " + \
                        "list-averaged surprisal on second relative to first lists. Error bars denote 95\% confidence intervals around "\
                        "the observed median (bootstrap estimate). "\
                        "The set-size and the length of intervening text are fixed at 10, and 435 tokens, respectively.")

    # now save as .tex file
    fname = os.path.join(savedir, tablefname + ".tex")
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
    print(args)

    p, stat = generate_plot(model=args.model)

    figsavename = f"context-structure_{tags[args.model]}_{model_ids[args.model]}"
    tablesavename = f"context-structure_{tags[args.model]}_{model_ids[args.model]}_table"

    savefig(p, savedir=args.savedir, figfname=figsavename)
    savetable(stat, savedir=args.savedir, tablefname=tablesavename)


if __name__ == "__main__":

    main()