import os
import pandas as pd
from func import make_timecourse_plot
from utils import load_csv_data, data_dir, savedir, table_savedir
import logging
import numpy as np
from matplotlib import pyplot as plt

# define dicts that contain text elements for plots
model_ids = {"gpt2": "a-10", "awd_lstm": "a-10", "w-12v2": "w-12v2"}
tags = {"gpt2": "gpt2", "awd_lstm": "awd_lstm", "w-12v2": "trf-w12v2"}
titles = {"gpt2": "Transformer (Radford et al, 2019)", "w-12v2": "Transformer (WT103, 12-layer)", "awd_lstm": "LSTM (Merity et al, 2017)"}
arcs = {"gpt2": "gpt2", "w-12v2": "gpt2", "awd_lstm": "awd_lstm"}
ylims={"gpt2": (0, None), "awd_lstm": (0, None), "w-12v2": (0, None)}
scenario = "sce1"


def generate_plot(model: str):

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
        data = load_csv_data(model=model, datadir=os.path.join(data_dir, "wt103_v2"), fname=f"*{model}*sce1_1_*.csv").dropna()
        data["model"] = "gpt2"
        data.prompt_len = data.prompt_len.map({1: 8, 2: 30, 3:100, 4:200, 5:400})
        data.context = data.context.map({"sce1": "intact"})


    # rename some row variables for plotting
    new_list_names = {"categorized": "semantic", "random": "arbitrary"}
    data.list = data.list.map(new_list_names)

    new_second_list_names = {"control": "novel", "repeat": "repeated", "permute": "permuted"}
    data.second_list = data.second_list.map(new_second_list_names)  


    # ===== SELECT DATA ===== #

    context_len = 8
    list_len = 10
    context = "intact"

    sel = (data.prompt_len == context_len) & \
          (data.list_len == list_len) & \
          (data.list.isin(["semantic", "arbitrary"])) & \
          (data.context == context) & \
          (data.model_id.isin([model_ids[model]])) & \
          (data.marker.isin([2, 3])) & \
          (data.second_list.isin(["repeated", "permuted", "novel"])) &\
          (data.marker_pos_rel.isin(list(range(-4, 10))))

    d = data.loc[sel].copy()

    # name column manually
    d.rename(columns={"list": "list structure", "second_list": "condition"}, inplace=True)
    # capitalize row values
    new_second_list_names = {"novel": "Novel", "repeated": "Repeated", "permuted": "Permuted"}
    d.condition = d.condition.map(new_second_list_names)

    # common fig properties
    w, h = 12, 2

    model_id = model_ids[model]
    suptitle = titles[model]
    arc = arcs[model]
    ylim = ylims[model]
        
    sel = ((d["model_id"] == model_id) & (d.model == arc))
    
    print(sum(sel))

    d.rename(columns={"marker_pos_rel": "marker-pos-rel", "condition": "Second list"}, inplace=True)
    
    p, ax, _ = make_timecourse_plot(d.loc[sel], x="marker-pos-rel", style="Second list", col="list structure",
                                    estimator=np.median,
                                    col_order=["arbitrary", "semantic"], err_style="band", 
                                    hue_order=["Repeated", "Permuted", "Novel"],
                                    style_order=["Repeated", "Permuted", "Novel"])

    _, _, stat = make_timecourse_plot(d.loc[sel], x="marker-pos-rel", style="Second list", col="list structure", 
                                    estimator = np.median,
                                    col_order=["arbitrary", "semantic"], err_style="bars", 
                                    hue_order=["Repeated", "Permuted", "Novel"],
                                    style_order=["Repeated", "Permuted", "Novel"])
    
    plt.close(plt.gcf())

    # set ylims
    ymin, ymax = ylim
    if ymin is None: ymin = ax[0].get_ylim()[0]
    if ymax is None: ymax = ax[0].get_ylim()[1]
    ax[0].set(ylim=(ymin, ymax))
    
    ax[0].set_title("Arbitrary list", fontsize=16)
    ax[1].set_title("Semantically coherent list", fontsize=16)
    ax[0].set_ylabel("Surprisal\n(bit)")
    for a in ax:
        a.set_xlabel("Token position relative to list onset")
        a.set_xticks(list(range(-4, 10, 2)))
        a.set_xticklabels(list(range(-4, 10, 2)))
            
    p.fig.suptitle(suptitle, fontsize=18)
    p.fig.set_size_inches(w=w, h=h)
    p.fig.subplots_adjust(top=0.65)

    return p, stat


def savefig(p, savedir: str, figfname: str):
            
    logging.info("Saving {}".format(os.path.join(savedir, figfname)))
    p.savefig(os.path.join(savedir, figfname + ".pdf"), transparent=True, bbox_inches="tight")
    p.savefig(os.path.join(savedir, figfname + ".png"), dpi=300, bbox_inches="tight")


def savetable(stat: pd.DataFrame, savedir: str, tablefname: str) -> None:
    
    # create a column with string formated and save the table as well
    stat = stat.round({"ci_min": 1, "ci_max": 1, "median": 1})
    strfunc = lambda x: str(x["median"]) + " " + "(" + str(x["ci_min"]) + "-" + str(x["ci_max"]) + ")"
    stat["report_str"] = stat.apply(strfunc, axis=1)

    # save the original .csv
    fname = os.path.join(savedir, tablefname + ".csv")
    print("Writing {}".format(fname))
    stat.to_csv(fname)

    # save for latex
    stat.rename(columns={"list structure": "List", "marker-pos-rel": "Token position"}, inplace=True)
    tex = stat.loc[stat["Token position"].isin(list(range(0, 4))), :]\
              .pivot(index=["List", "Second list"], columns=["Token position"], values="report_str")\
              .to_latex(bold_rows=True,
                        label=f"tab:{tablefname}",
                        caption="Surprisal values for four initial token positions, list type and second list condition.")

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
        args = parser.parse_args()

    p, stat = generate_plot(model=args.model)

    figsavename = f"timecourse_{scenario}_{tags[args.model]}_{model_ids[args.model]}"
    tablesavename = f"timecourse_{scenario}_{tags[args.model]}_{model_ids[args.model]}_table.tex"

    savefig(p, savedir=args.savedir, figfname=figsavename)
    savetable(stat, savedir=args.savedir, tablefname=tablesavename)


if __name__ == "__main__":

    main()
