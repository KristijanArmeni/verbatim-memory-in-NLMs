import os
import json

checkpoint_dir = "/scratch/ka2773/project/lm-mem/checkpoints"

tokenizers = {"pretrained": "gpt2", 
              "wikitext103": "/home/ka2773/project/lm-mem/data/wikitext-103_tokenizer"}

def load_trainer_state_json(path):
    
    print("Loading {}".format(path))
    with open(path, "r") as fhandle:
        trainer_state = json.load(fhandle)

    return trainer_state

w12_ckp = load_trainer_state_json(os.path.join(checkpoint_dir, "gpt2_40m_12-768-1024_a_02/checkpoint-31000/trainer_state.json"))
w06_ckp = load_trainer_state_json(os.path.join(checkpoint_dir, "gpt2_40m_6-768-1024_a_02/checkpoint-25000/trainer_state.json"))
w03_ckp = load_trainer_state_json(os.path.join(checkpoint_dir, "gpt2_40m_3-768-1024_a_02/checkpoint-19500/trainer_state.json"))
w01_ckp = load_trainer_state_json(os.path.join(checkpoint_dir, "gpt2_40m_1-768-1024_a_02/checkpoint-17000/trainer_state.json"))

id_to_type = {
        "a-10": ["pretrained", "gpt2", tokenizers["pretrained"]],
        "r-10": ["random", "", tokenizers["pretrained"]],
        "r-20": ["random-att", "gpt2", tokenizers["pretrained"]],
        "r-25": ["random-att-per-head", "gpt2", tokenizers["pretrained"]],
        "r-30": ["shuff-wpe", "gpt2", tokenizers["pretrained"]],
        "w-12": ["wikitext-12-layer", w12_ckp["best_model_checkpoint"], tokenizers["wikitext103"]],
        "w-06": ["wikitext-06-layer", w06_ckp["best_model_checkpoint"], tokenizers["wikitext103"]],
        "w-03": ["wikitext-03-layer", w03_ckp["best_model_checkpoint"], tokenizers["wikitext103"]],
        "w-01": ["wikitext-01-layer", w01_ckp["best_model_checkpoint"], tokenizers["wikitext103"]],
        }

savename = "/home/ka2773/project/lm-mem/src/greene_scripts/wm_eval/checkpoint_configs.json"
with open(savename, 'w') as fhandle:
    
    print("Writing {}".format(savename))
    json.dump(id_to_type, fhandle, indent=4)
    
