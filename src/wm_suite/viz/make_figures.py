
import os
import logging
import argparse
from wm_suite.viz import timecourse_figure
from wm_suite.viz import list_length_figure
from wm_suite.viz import context_length_figure
from wm_suite.viz import context_structure_figure
from wm_suite.viz import layer_figure
from wm_suite.viz import vignette_control_figure
import logging
from wm_suite.viz.func import set_manuscript_style


def main(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt2", "awd_lstm", "w-12v2"])
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    #set_manuscript_style()

    ##### ===== TIME COURSE PLOT ===== #####
    logging.info(f"Making timecourse figure for args.model {args.model}")
    timecourse_figure.main(["--model", args.model, "--savedir", args.savedir])


    ##### ===== LIST LENGTH PLOT ===== #####
    logging.info(f"Making list length figure for model {args.model}")
    list_length_figure.main(["--model", args.model, "--savedir", args.savedir])


    ##### ===== CONTEXT LENGTH PLOT ===== #####
    logging.info(f"Making context length figure for model {args.model}")
    context_length_figure.main(["--model", args.model, "--savedir", args.savedir])


    ##### ===== CONTEXT STRUCTURE PLOT ===== #####
    logging.info(f"Making context structure figure for model {args.model}")
    context_structure_figure.main(["--model", args.model, "--savedir", args.savedir])


    ##### ===== LAYER PLOT ===== #####
    for model_id in ["w-01v2", "w-03v2", "w-06v2", "w-12v2"]:

        logging.info(f"Making layer figure for model {model_id}")
        layer_figure.main(["--model_id", model_id, "--savedir", args.savedir])


    ##### ===== CONTROL VIGNETTES ===== #####
    for exp in ["punctuation", "name", "shuffled-preface", "shuffled-prompt"]:

        logging.info(f"Making control figure ({exp}) for model {args.model}")
        vignette_control_figure.main(["--model", f"{args.model}", "--experiment", exp, "--savedir", args.savedir])

    logging.info("\n##### ===== DONE ===== #####")

    return 0


if __name__ == "__main__":

    main()


