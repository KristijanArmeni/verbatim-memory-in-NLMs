import os
import pandas as pd
import logging
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

logging.basicConfig(level=logging.INFO)

#home_dir = os.path.join(os.environ['homepath'], "project", "lm-mem")
#data_dir = os.path.join(home_dir, "data", "outputs")
#savedir = os.path.join(home_dir, "fig", "raw", "camera_ready")
data_dir = "/home/ka2773/project/lm-mem/src/data/test_d"
#table_savedir = os.path.join(home_dir, "tables", "revised")


def load_wt103_data(path: str) -> pd.DataFrame:
    
    o = []
    files = glob(path)
    for f in tqdm(files, desc="file"):

        # temp: read in column values form filenames
        model_id = os.path.basename(f).split("_")[2]
        
        tmp = pd.read_csv(f, sep='\t')
        tmp['model_id'] = model_id
        tmp.token = tmp.token.str.strip("Ġ")
        tmp.rename(columns={"scenario": "context", "trialID": "marker", "stimID": "stimid"}, inplace=True)
        tmp.stimid = (tmp.stimid - 1)
        
        o.append(tmp)

    return pd.concat(o, ignore_index=True)    


def load_csv_data(model:str, datadir:str, fname:str) -> pd.DataFrame:

    if model == "gpt2":

        filename = os.path.join(datadir, fname)
        logging.info(f"Loading {filename}")
        data = pd.read_csv(filename, sep="\t", index_col=0)
    
    elif model == "awd_lstm":
    
        fn = os.path.join(datadir, fname)
        logging.info(f"Loading {fn}")
        data = pd.read_csv(fn, sep="\t", index_col=None)

    elif model in ["w-01v2", "w-03v2", "w-06v2", "w-12v2"]:

        logging.info(f"Loading matches to {fname}...dd")
        data = load_wt103_data(os.path.join(datadir, fname))
    
    return data