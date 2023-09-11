
import configparser
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(funcname)s() | %(message)s")

def get_filenames(pyfilename: str) -> Dict:

    # read filenames from pathsconfig.toml
    cfg = configparser.ConfigParser()
    logging.info(f"Reading entries in datapaths.toml for {pyfilename}")
    cfg.read("datapaths.toml")

    fnames = {**cfg[pyfilename]}

    return fnames
