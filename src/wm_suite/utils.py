import requests, zipfile, io
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Armeni et al, 2022
OSF_FILES_URL = "https://osf.io/5gy7x/files/osfstorage"
GPT2_RESULTS_OSF_URL = "https://osf.io/download/s3zv9/"
WT103_TRANSFORMER_OSF_URL = "https://osf.io/download/ejwug/"
AWD_LSTM_RESULTS_OSF_URL = "https://osf.io/download/yu7m3/"

# Armeni et al, 2023
OSF_URL_ATTN_WEIGHTS_ZIP_TMP = "https://osf.io/download/pm6wq/"  # temporary link on 


def download_raw_data_zip(zipurl: str, path: str):

    logging.info(f"Fetching data from {zipurl} ...")
    r = requests.get(zipurl)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    logging.info(f"Extracting .zip to {path} ...")
    z.extractall(path)

    logging.info("Done!")

    return 0


def download_raw_data_nonzip(url: str, path: str):

    myfile = requests.get(url)
    

def get_data(which: str, path: str):

    logging.info(f"Downloading .zip containing {which} data...")

    if which == "wt103_transformer":

        download_raw_data_zip(zipurl=WT103_TRANSFORMER_OSF_URL, path=path)

    elif which == "gpt2":

        download_raw_data_zip(zipurl=GPT2_RESULTS_OSF_URL, path=path)

    elif which == "awd_lstm":

        download_raw_data_zip(zipurl=AWD_LSTM_RESULTS_OSF_URL, path=path)

    elif which == "attention_weights":

        download_raw_data_zip(zipurl=OSF_URL_ATTN_WEIGHTS_ZIP_TMP, path=path)


def download_data():

    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, choices=["all", "gpt2", "awd_lstm", "wt103_transformer"])
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


# download the data for Armeni et al, 2023
def download_data2():

    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, choices=["attention_weights", "ablation_experiment"], required=True)
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    get_data(which=args.which, path=args.path)

    return None