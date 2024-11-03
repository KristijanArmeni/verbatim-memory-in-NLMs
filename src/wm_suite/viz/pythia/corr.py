from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, bootstrap
from wm_suite.utils import logger
from wm_suite.viz.pythia.mmlu_categories import categories, subcategories
from itertools import product
from tqdm import tqdm
from typing import Tuple

MODELS = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
TIMESTEPS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    3000,
    13000,
    23000,
    33000,
    43000,
    53000,
    63000,
    73000,
    83000,
    93000,
    103000,
    113000,
    123000,
    133000,
    143000
]

EVALS_PATH = Path("c://users/karmeni1/project/lm-mem/data/pythia_evals")
MEMPATH = Path("c://users/karmeni1/project/lm-mem/data/pythia")

# get dict that maps from subcategory to category
subcat2cat = {v: k for k, vals in categories.items() for v in vals}


def load_pythia_evals():

    model_step_combs = list(product(MODELS, TIMESTEPS))

    step_rows, mod_rows, task_rows = [], [], []
    acc_list = []
    for c in model_step_combs:

        mod, st = c # get model and step strings
        jsonfn = Path(EVALS_PATH, f"{mod}_step{st}.json")

        if jsonfn.exists():
            with open(jsonfn, "r") as f:
                all_tasks_dict = json.load(f)['results']

            for task in list(all_tasks_dict.keys()):
                task_dict = all_tasks_dict[task]

                if "acc" in task_dict.keys():   
                    step_rows.append(st)
                    mod_rows.append(mod)
                    task_rows.append(task)
                    acc_list.append(task_dict['acc'])
                else:
                    pass
        else:
            print(f"{jsonfn} does not exist")

    df = pd.DataFrame({
        "step": step_rows,
        "model": mod_rows,
        "task": task_rows,
        "acc": acc_list
    })
    
    # add MMLU category and subcategory columns
    df['task'] = df.task.str.replace("hendrycksTest-", "mmlu_")
    df['benchmark'] = df.task.map(lambda x: "mmlu" if "mmlu" in x else x)
    df["mmlu_subcategory"] = df.task.map(lambda x: subcategories[x.replace("mmlu_", "")][0] if "mmlu" in x else x)
    df["category"] = df.mmlu_subcategory.map(lambda x: "mmlu-"+subcat2cat[x] if x in subcat2cat.keys() else x)

    return df


def load_timecourse_data() -> dict:

    ckp_steps_dict = {

        "pythia-160m": TIMESTEPS,
        "pythia-410m": TIMESTEPS,
        "pythia-1.4b": TIMESTEPS,
        "pythia-2.8b": TIMESTEPS,
        "pythia-6.9b": TIMESTEPS,
        "pythia-12b": TIMESTEPS,
    }

    mod_list, step_list, mem_list = [], [], []

    for ckp, ckp_steps in tqdm(ckp_steps_dict.items(), desc='checkpoint'):

        suffix = "repeat_mem.json"
        
        ckp_folder = ckp.replace("-", "_")

        for step in ckp_steps:

            fn = Path(MEMPATH, ckp_folder, f"{ckp}_step{step}_{suffix}")
            with open(fn, 'r') as f:
                mem = json.load(f)
            
            mod_list.append(ckp.strip("pythia-"))
            step_list.append(step)
            mem_list.append(np.mean(np.mean(mem["rs"], axis=0), axis=0))

    df = pd.DataFrame({
        "model": mod_list,
        "step": step_list,
        "mem": mem_list
    })

    return df 


def min_max(arr):
    return (arr - arr.min())/(arr.max() - arr.min())

def fetch_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    df1 = load_pythia_evals()
    df2 = load_timecourse_data()

    return df1, df2

def spearmanr_wrapper(x, y):

    rho, _ = spearmanr(x, y)
    return rho


def learning_traj_correlations(d1: pd.DataFrame, d2:pd.DataFrame) -> pd.DataFrame:

    # average across mmlu subcategories
    d1 = d1.groupby(['step', 'model', 'benchmark', 'category'])[['acc']].mean().reset_index() 

    # find models and tasks that are common to both dataframes
    models = set(d1.model).intersection(set(d2.model))
    tasks = d1.category.unique()

    combs = list(product(models, tasks))
    out = {
        "model": [],
        "benchmark": [],
        "rho": [],
        "final_acc": [],
        "CI_low": [],
        "CI_high": []
    }

    for c in tqdm(combs, desc="Correlation"):

        mod, cat = c # model and benchmark task string

        # benchmark performance
        vals1 = d1[(d1.model == mod) & (d1.category == cat)].acc
        vals2 = d2[d2.model == mod].mem  # retrieval performance

        assert len(vals1) == len(vals2), f"Length mismatch for {mod} and {cat} ({len(vals1)} vs {len(vals2)})"

        # compute rank-based correlation
        rho = spearmanr_wrapper(vals1, vals2)

        # estmiate confidence intervals
        ci = bootstrap(
            data=(vals1, vals2),
            statistic=spearmanr_wrapper,
            n_resamples=5000,
            paired=True,
            vectorized=False,
            method='BCa',
        ).confidence_interval

        # stor benchmark accuracy for fully-trained model
        selrow = (d1.model == mod) & (d1.category == cat) & (d1.step == TIMESTEPS[-1])
        final_acc = d1[selrow].acc.to_numpy().item()

        out["model"].append(mod)
        out["benchmark"].append(cat)
        out["rho"].append(rho)
        out["final_acc"].append(final_acc)
        out['CI_low'].append(ci.low)
        out['CI_high'].append(ci.high)

    corr_df = pd.DataFrame(out)

    return corr_df



def main():

    import argparse

    parser = argparse.ArgumentParser(description="Compute learning trajectory correlations")
    
    parser.add_argument("--savedir", type=str, required=True, help="Directory to save output")
    args = parser.parse_args()

    benchmark_df, retrieval_df = fetch_data()
    corr_df = learning_traj_correlations(benchmark_df, retrieval_df)

    savedir = Path(args.savedir)
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)

    savefn = Path(savedir, "learning_traj_corr.tsv")
    logger.info(f"Saving learning trajectory correlations to {savefn}")
    corr_df.to_csv(savefn, sep="\t", index=False)
    

if __name__ == '__main__':

    main()