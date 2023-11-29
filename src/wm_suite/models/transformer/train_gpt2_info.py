import os
import json
import torch
import logging
import pandas as pd
from transformers import GPT2LMHeadModel

logging.basicConfig(format="[INFO] %(message)s", level=logging.INFO)

# define dataframe columns and rows
cols = ["w-01", "w-03", "w-06", "w-12"]
model_params = ["activation_function", "n_layer", "n_head", "n_ctx", "n_positions", "vocab_size"]
train_params = ["per_device_train_batch_size", "per_device_eval_batch_size", "learning_rate", "adam_beta1", "adam_beta2"]

rows = model_params + train_params
# create data frame storing checkpoin info in columns and param values as rows
df = pd.DataFrame(columns=cols, index=rows)

# load in the file containing checkpoints used
cfg_json_path = os.path.join(os.environ["HOME"], 'project/lm-mem/src/greene_scripts/wm_eval/checkpoint_configs.json')

with open(cfg_json_path, "r") as fhandle:
    ckp_cfg = json.load(fhandle)

# loop over checkpoints used
for ckp_key in cols:

    checkpoint_path = ckp_cfg[ckp_key][1]

    logging.info("Working on {}".format(checkpoint_path))

    model_config_path = os.path.join(checkpoint_path, "config.json")
    training_args_path = os.path.join(checkpoint_path, "training_args.bin")

    with open(model_config_path, "r") as fhandle:
        model_cfg = json.load(fhandle)

    # load the TrainingArguments class
    train_args = vars(torch.load(training_args_path))

    # collect model parameters
    for param_key in model_params:
        df.loc[param_key, ckp_key] = model_cfg[param_key]

    for param_key in train_params:
        df.loc[param_key, ckp_key] = train_args[param_key]

    # get number of parameters
    model = GPT2LMHeadModel.from_pretrained(os.path.join(checkpoint_path))
    n_params = model.num_parameters(only_trainable=True)

    df.loc["n params (M)", ckp_key] = round(n_params/1e6, 1)

# rename some rows
new_rownames = {key: key.replace("_", " ") for key in rows}
new_colnames = {key: str(int(key.split("-")[-1])) + " layer" for key in cols}
df.rename(index=new_rownames, columns=new_colnames, inplace=True)
df.rename(index={"n ctx": "n context (tokens)", "n positions": "n context (tokens)"})

# convert df to .tex
logging.info("First 15 rows of df:")
display(df.head(15))

tex = df.to_latex(bold_rows=True,
                  label="tab:model-train-info",
                  caption="Architecture and model parameters.")

fname = os.path.join(os.environ['HOME'], "project/lm-mem/src/greene_scripts/model_train_params.tex")
logging.info("Writing {}".format(fname))

with open(fname, "w") as fhandle:
    fhandle.writelines(tex)

