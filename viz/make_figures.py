
import os
import logging
import timecourse_figure
import list_length_figure
import context_length_figure
import context_structure_figure
import layer_figure
import vignette_control_figure
import logging
from func import set_manuscript_style


def main(model, figpath):

    set_manuscript_style()

    ##### ===== TIME COURSE PLOT ===== #####
    logging.info(f"Making timecourse figure for model {model}")
    timecourse_figure.main(["--model", model, "--savedir", figpath])


    ##### ===== LIST LENGTH PLOT ===== #####
    logging.info(f"Making list length figure for model {model}")
    list_length_figure.main(["--model", model, "--savedir", figpath])


    ##### ===== CONTEXT LENGTH PLOT ===== #####
    logging.info(f"Making context length figure for model {model}")
    context_length_figure.main(["--model", model, "--savedir", figpath])


    ##### ===== CONTEXT STRUCTURE PLOT ===== #####
    logging.info(f"Making context structure figure for model {model}")
    context_structure_figure.main(["--model", model, "--savedir", figpath])


    ##### ===== LAYER PLOT ===== #####
    for model_id in ["w-01v2", "w-03v2", "w-06v2", "w-12v2"]:

        logging.info(f"Making layer figure for model {model_id}")
        layer_figure.main(["--model_id", model_id, "--savedir", figpath])


    ##### ===== CONTROL VIGNETTES ===== #####
    for exp in ["punctuation", "name", "shuffled-preface", "shuffled-prompt"]:

        logging.info(f"Making control figure ({exp}) for model {model}")
        vignette_control_figure.main(["--model", f"{model}", "--experiment", exp, "--savedir", figpath])


    return 0


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--savedir", type=str)

    args = parser.parse_args()

    main(args.model, args.savedir)


