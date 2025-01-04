import argparse
from wm_suite.utils import download_raw_data_zip

# Armeni et al, 2024, CoNNLL
OSF_CONLL24_EXP_1 = "https://osf.io/download/mncwt/"
OSF_CONLL24_EXP_2 = "https://osf.io/download/afhsp/"
OSF_CONLL24_EXP_3 = "https://osf.io/download/rdfs8/"
OSF_CONLL24_EXP_1_ZIPFNAME = "01_exp_retrieval-x-time"
OSF_CONLL24_EXP_2_ZIPFNAME = "02_exp_benchmark-correlations"
OSF_CONLL24_EXP_3_ZIPFNAME = "03_exp_concrete-abstract"

def download_data_conll24():

    parser = argparse.ArgumentParser("Downloads data from 10.17605/OSF.IO/A6GSW")
    parser.add_argument(
        "which",
        type=str,
        choices=["all", "exp1", "exp2"],
    )
    parser.add_argument(
        "savepath",
        type=str,
    )

    args = parser.parse_args()

    arg2url = {
        "exp1": OSF_CONLL24_EXP_1,
        "exp2": OSF_CONLL24_EXP_2,
        "exp3": OSF_CONLL24_EXP_3
    }

    arg2fname = {
        "exp1": OSF_CONLL24_EXP_1_ZIPFNAME,
        "exp2": OSF_CONLL24_EXP_2_ZIPFNAME,
        "exp3": OSF_CONLL24_EXP_3_ZIPFNAME,
    }

    if args.which == "all":
        for expkey, zipurl in arg2url.items():
            download_raw_data_zip(zipurl=zipurl, path=args.savepath + "/" + arg2fname[expkey])
    else:
        download_raw_data_zip(zipurl=arg2url[args.which], path=args.savepath + "/" + arg2fname[args.which])

    return None


if __name__ == "__main__":

    download_data_conll24()