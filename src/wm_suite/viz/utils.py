import os
import yaml
import pandas as pd
import logging
from glob import glob
from tqdm import tqdm
from types import SimpleNamespace
from ..utils import logger


# home_dir = os.path.join(os.environ['homepath'], "project", "lm-mem")
# data_dir = os.path.join(home_dir, "data", "outputs")
# savedir = os.path.join(home_dir, "fig", "raw", "camera_ready")
data_dir = "/home/ka2773/project/lm-mem/src/data/test_d"
# table_savedir = os.path.join(home_dir, "tables", "revised")


clrs = SimpleNamespace(
    **{
        "green": "#2ca02c",
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
    }
)


def get_font_config(pymodule, currentfont):
    with open(os.path.join(os.path.dirname(__file__), "fontconfig.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    if currentfont == "DejaVu Sans":
        fontkey = "dejavusans"
    elif currentfont == "Segoe UI":
        fontkey = "segoeui"
    else:
        raise ValueError(f"Don't have config for {currentfont} in fontconfig.yaml")

    fontcfg = cfg[pymodule][fontkey]
    print_names = "".join([f"- {k}: {v}\n" for k, v in fontcfg.items()])
    logger.info(f"Specified font size config for {currentfont}:\n{print_names}")

    return fontcfg


def load_wt103_data(path: str) -> pd.DataFrame:
    o = []
    files = glob(path)

    logger.info(f"Found {len(files)} files...")

    for f in tqdm(files, desc="file"):
        # temp: read in column values form filenames
        model_id = os.path.basename(f).split("_")[2]

        tmp = pd.read_csv(f, sep="\t")
        tmp["model_id"] = model_id
        tmp.token = tmp.token.str.strip("Ġ")
        tmp.rename(
            columns={"scenario": "context", "trialID": "marker", "stimID": "stimid"},
            inplace=True,
        )
        tmp.stimid = tmp.stimid - 1

        o.append(tmp)

    return pd.concat(o, ignore_index=True)


def load_csv_data(model: str, datadir: str, fname: str) -> pd.DataFrame:
    if model == "gpt2":
        filename = os.path.join(datadir, fname)
        logger.info(f"Loading {filename}")
        data = pd.read_csv(filename, sep="\t", index_col=0)

    elif model == "awd_lstm":
        fn = os.path.join(datadir, fname)
        logger.info(f"Loading {fn}")
        data = pd.read_csv(fn, sep="\t", index_col=None)

    elif model in ["w-01v2", "w-03v2", "w-06v2", "w-12v2"]:
        logger.info(f"Loading matches to {fname}...")
        data = load_wt103_data(os.path.join(datadir, fname))

    return data


def save_png_pdf(fig, savename: str):
    savefn = os.path.join(savename + ".png")
    logger.info(f"Saving {savefn}")
    fig.savefig(savefn, dpi=300, format="png", bbox_inches="tight")

    savefn = os.path.join(savename + ".pdf")
    logger.info(f"Saving {savefn}")
    fig.savefig(savefn, format="pdf", transparent=True, bbox_inches="tight")

    return 0
