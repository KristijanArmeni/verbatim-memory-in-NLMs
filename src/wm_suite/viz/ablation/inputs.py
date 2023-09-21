
import configparser
from typing import Dict
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(funcname)s() | %(message)s")

def get_filenames(pyfilename: str) -> Dict:

    # read filenames from pathsconfig.toml
    cfg = configparser.ConfigParser()
    logging.info(f"Reading entries in inputs.yaml for {pyfilename}")
    #cfg.read("datapaths.toml")

    with open("inputs.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    fnames = cfg[pyfilename]
    print_names = "".join([f'- {k}: {v}\n' for k, v in fnames.items()])
    logging.info(f"Specified module input files:\n{print_names}")

    return fnames
