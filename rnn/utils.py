import os
from typing import Dict
import json
import sys
from datetime import datetime

def get_configs_for_dev(config: str) -> Dict:

    if "win" in sys.platform:
        project_root = os.path.join(os.environ['homepath'], 'project', 'lm-mem')
        checkpoint_dir = "C:\\Users\\karmeni1\\project\\lm-mem\\data\\checkpoints\\lstm\\train-test"
        log_dir = "C:\\Users\\karmeni1\\project\\lm-mem\\data\\logs\\lstm\\"

    elif "linux" in sys.platform:
        project_root = os.path.join(os.environ['HOME'], 'project', 'lm-mem')
        checkpoint_dir = "/scratch/ka2773/project/lm-mem/checkpoints/lstm"
        log_dir = "/scratch/ka2773/project/lm-mem/checkpoints/lstm"

    model_config = {

        "n_vocab": 28439,
        "n_inp": 50,
        "n_hid": 50,
        "n_layers": 4,
        "nonlinearity": "relu",
        "dropout": 0.1,
        "truncated_bptt_steps": 20, 
        "example_input_array": False,

    }

    # set up dataset configuration
    data_config = { "datadir": os.path.join(project_root, "data", "wikitext-103"),
                    "vocab_path": os.path.join(project_root, "src", "rnn", "vocab.txt"),
                    "train_bs": 16, 
                    "valid_bs": 16, 
                    "test_bs": 5, 
                    "num_workers": 4,
                    "train_size": 40e6,
                    "per_batch_seq_len": 200,  # sequence len per batch, this is in memory for forward pass
                    "bptt_len": 50             # detach gradients every 20 tokens, pl.Trainer takes care of this
                    }

    now = datetime.now().strftime("%H-%M-%S")

    trainer_config = {
        "root_dir": checkpoint_dir,
        "log_dir": log_dir,
        "wandb_project": "",
        "wandb_group": "test-run",
        "wandb_name": "test-name",
        "wandb_id": f"test-id-{now}"
    }

    if config == "model_config":
        return_config = model_config 
    elif config == "data_config":
        return_config = data_config
    elif config == "trainer_config":
        return_config = trainer_config
    
    return return_config


def load_json_config(config_path: str) -> Dict:
    
    with open(config_path, "r") as fh:
        json_config = json.load(fh)

    return json_config