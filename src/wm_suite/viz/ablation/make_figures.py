
import argparse
import ablation_figure, attn_weights_per_layer_figure
from viz.func import set_manuscript_style
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    set_manuscript_style()

    logging.info("Drawing ablation plot...")
    ablation_figure.main(["--datadir", args.datadir, "--savedir", args.savedir])

    logging.info("Drawing attention weights figure...")
    attn_weights_per_layer_figure.main(["--datadir", args.datadir, "--savedir", args.savedir])


    return 0

if __name__ == "__main__":

    main()