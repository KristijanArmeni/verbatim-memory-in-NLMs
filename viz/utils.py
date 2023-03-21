import os
import pandas as pd
import logging
from glob import glob
from tqdm import tqdm

home_dir = os.path.join(os.environ['homepath'], "project", "lm-mem")
data_dir = os.path.join(home_dir, "data", "outputs")
savedir = os.path.join(home_dir, "fig", "raw", "camera_ready")
table_savedir = os.path.join(home_dir, "tables", "revised")

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

    elif model == "w-12v2":

        data = load_wt103_data(os.path.join(datadir, fname))
    
    return data


