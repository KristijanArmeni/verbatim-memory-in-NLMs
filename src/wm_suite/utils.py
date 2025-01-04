import requests, zipfile, io
import logging
import argparse
import torch

# Armeni et al, 2022
OSF_FILES_URL = "https://osf.io/5gy7x/files/osfstorage"
GPT2_RESULTS_OSF_URL = "https://osf.io/download/s3zv9/"
WT103_TRANSFORMER_OSF_URL = "https://osf.io/download/ejwug/"
AWD_LSTM_RESULTS_OSF_URL = "https://osf.io/download/yu7m3/"

# Armeni et al, 2023 (TEMPORARY LINKS)
OSF_URL_ATTN_WEIGHTS_ZIP_TMP = "https://osf.io/download/pm6wq/"  # temporary link on
OSF_URL_SVA_EXP_ZIP_TMP = "https://osf.io/download/jgdt6/"
OSF_URL_WT103_EXP = "https://osf.io/download/db3ht/"

def set_cuda_if_available(device: str="cuda" or "cpu") -> torch.device:
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU!")
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    return device


def get_wm_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        "[%(levelname)s] %(module)s.%(funcName)s() | %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = get_wm_logger("wm_suite.utils")


def set_logger_level(level: str) -> logging.Logger:
    logger = logging.getLogger("wm_suite.utils")

    if level == "info":
        logger.setLevel(logging.INFO)
    elif level == "critical":
        logger.setLevel(logging.CRITICAL)


def download_raw_data_zip(zipurl: str, path: str):
    logger.info(f"Fetching data from {zipurl} ...")
    r = requests.get(zipurl)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    logger.info(f"Extracting .zip to {path} ...")
    z.extractall(path)

    logging.info("Done!")

    return 0


def download_raw_data_nonzip(url: str, path: str):
    myfile = requests.get(url)


def get_data(which: str, path: str):
    logging.info(f"Downloading .zip containing {which} data...")

    if which == "all":
        all_data = [
            GPT2_RESULTS_OSF_URL,
            WT103_TRANSFORMER_OSF_URL,
            AWD_LSTM_RESULTS_OSF_URL,
            OSF_URL_ATTN_WEIGHTS_ZIP_TMP,
        ]

        for zip_url in all_data:
            download_raw_data_zip(zipurl=zip_url, path=path)

    elif which == "wt103_transformer":
        download_raw_data_zip(zipurl=WT103_TRANSFORMER_OSF_URL, path=path)

    elif which == "gpt2":
        download_raw_data_zip(zipurl=GPT2_RESULTS_OSF_URL, path=path)

    elif which == "awd_lstm":
        download_raw_data_zip(zipurl=AWD_LSTM_RESULTS_OSF_URL, path=path)

    elif which == "attention_weights":
        download_raw_data_zip(zipurl=OSF_URL_ATTN_WEIGHTS_ZIP_TMP, path=path)


def download_data():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which", type=str, choices=["all", "gpt2", "awd_lstm", "wt103_transformer"]
    )
    parser.add_argument("--path", type=str)

    args = parser.parse_args()

    # download a single model or download all if no flag is provided
    if args.which:
        get_data(which=args.which, path=args.path)
    else:
        get_data(which="gpt2", path=args.path)
        get_data(which="awd_lstm", path=args.path)
        get_data(which="wt103_transformer", path=args.path)

    return None


