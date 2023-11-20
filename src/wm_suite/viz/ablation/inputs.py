import os
from typing import Dict
from wm_suite.utils import logger
import yaml


def get_filenames(pyfilename: str) -> Dict:
    logger.info(f"Reading entries in inputs.yaml for {pyfilename}")

    with open(os.path.join(os.path.dirname(__file__), "inputs.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    fnames = cfg[pyfilename]
    print_names = "".join([f"- {k}: {v}\n" for k, v in fnames.items()])
    logger.info(f"Specified module input files:\n{print_names}")

    return fnames
