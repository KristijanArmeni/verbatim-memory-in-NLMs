"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""
import os
import json
from itertools import product

# home directory on marcc compute cluster
# this is git versioned repo:
root_dir = os.path.expanduser("~/project/lm-mem/src")
scripts_dir = os.path.join(root_dir, "hpc", "wm_eval", "wt103")
log_dir = "/scratch/ka2773/project/lm-mem/logs/wm_eval/wt103"

# open the main file that will submit the jobs
master_bash = open(os.path.join(scripts_dir, 'run_wt103_surprisal_scripts.sh'), 'w')

# load .json dict with paths to best checkpoints
checkpoint_config_file = os.path.join(root_dir, "hpc", "wm_eval", "checkpoint_configs.json")
with open(checkpoint_config_file, 'r') as f:
    id_to_type = json.load(f)

model_ids = ["w-01v2"]
scenarios = ["sce3"]
list_lens = ["7", "10"]
prompt_lens = ["1"]
conditions = ["repeat",]
list_types = ["random"]

# create combination of variable values
variables = product(model_ids, scenarios, list_lens, prompt_lens, conditions, list_types)

# now create the scripts with all the input args etc.

for vars in list(variables):

    # unpack the tuple
    model_id, scenario, list_len, prompt_len, condition, list_type = vars
    
    outname = f"surprisal_trf_{model_id}_{scenario}_{prompt_len}_{list_len}_{condition}_{list_type}.csv"
    
    noun_list_file = os.path.join(f"/home/ka2773/project/lm-mem/src/data/noun_lists/{list_type}_lists.json")

    output_path = "/scratch/ka2773/project/lm-mem/output/wt103v2"
    python_file = os.path.join(root_dir, "src", "wm_suite", "wm_test_suite.py")

    # seed is only used with random/permuted models
    model_seed=12345
    
    model_type = "pretrained"

    # create command string
    command = f"python {python_file} \\\n" \
              f"--condition {condition} \\\n" \
              f"--scenario {scenario} \\\n" \
              f"--list_type {list_type} \\\n" \
              f"--list_len {list_len} \\\n" \
              f"--prompt_len {prompt_len} \\\n" \
              f"--noun_list_file {noun_list_file} \\\n" \
              f"--pretokenize_moses \\\n" \
              f"--model_type {model_type} \\\n" \
              f"--checkpoint {id_to_type[model_id][1]} \\\n" \
              f"--tokenizer {id_to_type[model_id][-1]} \\\n" \
              f"--model_seed {model_seed} \\\n" \
              f"--context_len {1024} \\\n" \
              f"--batch_size {10} \\\n" \
              f"--output_dir {output_path} \\\n" \
              f"--output_file {outname} \\\n" \
              f"--device cuda \\\n" \

    scr_filename = f"script_surp_trf_{model_id}_{scenario}_{list_len}_{prompt_len}_{condition}_{list_type}"

    f = open(os.path.join(scripts_dir, scr_filename) + '.scr', 'w')

    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=" + scr_filename + "\n")
    f.write("#SBATCH --time=01:00:00\n")
    f.write("#SBATCH --mem 5gb\n" )
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks-per-node=1\n")
    f.write("#SBATCH --cpus-per-task=6\n")
    f.write("#SBATCH --mail-type=end\n")
    f.write("#SBATCH --mail-user=karmeni1@jhu.edu\n")
    f.write("#SBATCH --output=" + os.path.join(log_dir, scr_filename) + ".log\n")
    f.write("#SBATCH --error=" + os.path.join(log_dir, scr_filename) + ".err\n\n\n")

    f.write("singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c \"\n")

    f.write("source /ext3/env.sh\n")

    f.write("conda activate core_env\n\n")     # load environment with pytorch 1.6
    f.write(command + "\"" + "\n\n")                  # write the python command to be executed
    f.close()

    print("Writing {}".format(scr_filename))
    master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()
