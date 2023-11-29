# %%
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wm_suite.viz.ablation.inputs import get_filenames
from wm_suite.viz.utils import save_png_pdf, get_font_config

# %% make_plot


def make_plot(datadir: str):
    files = get_filenames(os.path.basename(__file__))
    fontcfg = get_font_config(
        os.path.basename(__file__), currentfont=plt.rcParams["font.sans-serif"][0]
    )

    dat = pd.read_csv(os.path.join(datadir, files["targeted"]), sep="\t")
    dat2 = pd.read_csv(os.path.join(datadir, files["random"]), sep="\t")
    dat_unablated = pd.read_csv(os.path.join(datadir, files["unablated"]), sep="\t")
    dat_random = pd.read_csv(os.path.join(datadir, files["rand-init"]), sep="\t")
    dat_all = pd.read_csv(os.path.join(datadir, files["ablate-all"]), sep="\t")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    ax.plot(
        dat["step"],
        np.log(dat["ppl"]),
        "--o",
        label="Heads found in\nattention-informed search",
    )
    ax.plot(
        dat2["step"],
        np.log(dat2["ppl"]),
        "--^",
        label="Heads found in\nUnconstrained search",
    )

    ax.hlines(
        np.log(dat_unablated["ppl"]),
        1,
        len(dat["step"]),
        linestyle="--",
        color="black",
        label="Unablated model",
    )
    ax.hlines(
        np.log(dat_all["ppl"]),
        1,
        len(dat["step"]),
        linestyle="-.",
        color="tab:red",
        label="Fully ablated model\n(all heads)",
    )

    annot_str = (
        f"Randomly initialized model ppl: {np.log(dat_random['ppl'].item()):.1f}"
    )
    ax.text(3, 4.1, s=annot_str, fontsize=fontcfg["annotfs"])

    ax.set_title(
        "Does ablating heads important for verbatim retrieval affect next-word prediction?\n(performance on Wikitext103 test set)",
        fontsize=fontcfg["titlefs"],
    )

    ax.set_xlabel("Number of heads ablated", fontsize=fontcfg["labelfs"])
    ax.set_ylabel("$\log$(perplexity)", fontsize=fontcfg["labelfs"])

    ax.set_xticks(np.arange(1, len(dat["step"]) + 1))
    ax.set_xticklabels(
        ["" if i % 2 == 0 else i for i in np.arange(1, len(dat["step"]) + 1, 1)],
        fontsize=fontcfg["tickfs"],
    )

    ax.tick_params(axis="y", labelsize=fontcfg["tickfs"])

    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle="--", linewidth=0.5, color="tab:grey")

    plt.tight_layout()

    return fig, ax


# %%


def main(input_args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", type=str)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    fig, _ = make_plot(datadir=args.datadir)

    if args.savedir:
        savename = os.path.join(args.savedir, "fig_circuits_ppl")
        save_png_pdf(fig=fig, savename=savename)


# %%
if __name__ == "__main__":
    main()
