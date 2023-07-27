"""
main script invoking the .main() method in individual plotting scripts to generate figures
"""
import os
import argparse

# own modules
from paths import PATHS as p
from src.wm_suite.viz.ablation import (fig_attn_example, 
                                       fig_attn, 
                                       fig_triplets)

from src.wm_suite.viz.func import set_manuscript_style

import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--which", type=str, choices=["all", "attn_example", "attn_weights", "triplets"])
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    flags = {"all": False, "attn_example": False, "attn_weights": False, "triplets": False}

    if args.which == "all":
        flags["all"] = True
        flags["attn_example"] = True
        flags["attn_weights"] = True
        flags["triplets"] = True
    else:
        flags[args.which] = True

    set_manuscript_style()

    # if flags["attn_example"]:
        #logging.info("Drawing attention weights example plot...")
        #attn_weights_example.main(["--datadir", args.datadir, "--savedir", args.savedir])

    # if flags["attn_weights"]:
        #logging.info("Drawing attention weights figure...")
        #attn_weights_figure.main(["--datadir", args.datadir, "--which", "main_fig", "--savedir", args.savedir])

    if flags["triplets"]:

        logging.info("Drawing memory ablation figure...")
        if args.datadir is None:
            args.datadir = os.path.join(p.data, "ablation", "zero_attn", "topk", "triplets")
        if args.savedir is None:
            args.savedir = os.path.join(p.root, 'fig', "ablation", "topk", "triplets")

        fig_triplets.main(["--datadir", args.datadir, "--savedir", args.savedir])

    return 0

if __name__ == "__main__":

    main()