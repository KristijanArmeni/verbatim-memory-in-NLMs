import os
import numpy as np
from matplotlib import pyplot as plt
from wm_suite.utils import logger


def make_plot(datadir: str, pos: int = -1):
    
    logger.info(f"Loading {os.path.join(datadir, 'patching_avg.npy')}")

    data = np.load(
        os.path.join(datadir, "patching_avg.npy")
    )  # shape = (layers, pos, heads)

    fig, ax = plt.subplots(figsize=(5, 5))

    data = data * 100

    # find absmax for color normalization
    absmax = np.abs(np.round(data[:, pos, :])).max()

    im = ax.imshow(
        data[:, pos, :],
        cmap="RdBu",
        origin="lower",
        aspect="equal",
        vmin=-absmax,
        vmax=absmax,
    )

    cax = ax.inset_axes([1.04, 0, 0.05, 1])
    cbar = plt.colorbar(im, cax=cax)

    cbar.ax.set_ylabel(
        "Proportion logit diff. recovered (%)\n($\\frac{\mathrm{patch}(x^{corrupted}_\Delta)-x^{corrupted}_\Delta}{x^{orig}_\Delta-x^{corrupted}_\Delta}$)"
    )

    # plot values for top and bottom 2 percentiles
    tmp = data[:, pos, :]
    rows, cols = np.where(
        (tmp > np.percentile(tmp, 98)) | (tmp < np.percentile(tmp, 2))
    )

    for r, c in zip(rows, cols):
        fs = 8 if len(f"{data[r, pos, c]:.1f}") <= 4 else 6
        ax.text(
            c,
            r,
            f"{data[r, pos, c]:.1f}",
            fontsize=fs,
            fontweight="semibold",
            ha="center",
            va="center",
            color="white" if data[r, pos, c] < -5 else "black",
        )

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    ax.set_xticks(np.arange(0, 12, 1))
    ax.set_xticklabels([i if i % 2 != 0 else "" for i in range(1, 13, 1)])
    ax.set_yticks(np.arange(0, 12, 1))
    ax.set_yticklabels([i if i % 2 != 0 else "" for i in range(1, 13, 1)])

    ax.grid(visible=True, color="lightgray", linewidth=0.5, linestyle="--", alpha=0.7)

    if pos in [46, -1]:
        ax.set_title("Patching attn output on corrupted run\n(N1 prediction at `:`)")
    else:
        ax.set_title(
            f"Patching attn output on corrupted run\n(N1 prediction at pos {pos})"
        )

    plt.tight_layout()

    return fig, ax


def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default=None,
                        help="Path where path patching output is (*/patching_avg.npy)")
    parser.add_argument("--savedir", type=str, default=None,
                        help="Path where to save the figure")

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    #datadir = "/scratch/ka2773/project/lm-mem/output/patching"

    fig, _ = make_plot(args.datadir, pos=-1)

    #savedir = "/home/ka2773/project/lm-mem/data/fig"
    fn = "patching_avg_test.png"
    savename = os.path.join(args.savedir, fn)
    logger.info(f"Saving figure to {savename}")

    fig.savefig(savename, dpi=300)

    plt.close("all")


if __name__ == "__main__":
    main()
