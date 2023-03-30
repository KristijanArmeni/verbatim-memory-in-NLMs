import requests, zipfile, io
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')

OSF_FILES_URL = "https://osf.io/5gy7x/files/osfstorage"
GPT2_RESULTS_OSF_URL = "https://osf.io/download/s3zv9/"
WT103_TRANSFORMER_OSF_URL = "https://osf.io/download/ejwug/"
AWD_LSTM_RESULTS_OSF_URL = "https://osf.io/download/yu7m3/"


def download_raw_data_zip(zipurl: str, path: str):

    logging.info(f"Fetching data from {zipurl} ...")
    r = requests.get(zipurl)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    logging.info(f"Extracting .zip to {path} ...")
    z.extractall(path)

    logging.info("Done!")

    return 0


def get_data(model: str, path: str):

    logging.info(f"Downloading .zip for {model} from {OSF_FILES_URL}")

    if model == "wt103_transformer":

        download_raw_data_zip(zipurl=WT103_TRANSFORMER_OSF_URL, path=path)

    elif model == "gpt2":

        download_raw_data_zip(zipurl=GPT2_RESULTS_OSF_URL, path=path)

    elif model == "awd_lstm":

        download_raw_data_zip(zipurl=AWD_LSTM_RESULTS_OSF_URL, path=path)


def download_data():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt2", "awd_lstm", "wt103_transformer"])
    parser.add_argument("--path", type=str)

    args = parser.parse_args()

    get_data(model=args.model, path=args.path)

    return 0
