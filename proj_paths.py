import configparser
import logging
import sys
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()
config.read("./config.toml")

PATHS = SimpleNamespace(**config["paths"])

if __name__ == "__main__":
    logging.info(f"Adding {PATHS.src} to pythonpath")
    sys.path.append(PATHS.src)
