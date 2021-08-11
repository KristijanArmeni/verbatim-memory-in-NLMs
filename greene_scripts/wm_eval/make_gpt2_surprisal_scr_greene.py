"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""
import os

# home directory on marcc compute cluster
# this is git versioned repo:
root_dir = os.path.expanduser("~/project/lm-mem/src")
scripts_dir = os.path.join(root_dir, "greene_scripts", "wm_eval")
log_dir = "/scratch/ka2773/project/lm-mem/logs/wm_eval"

master_bash = open(os.path.join(scripts_dir, 'run_gpt2_surprisal_scripts.sh'), 'w')

# model labels
# a-10 | pretrained gpt-small

# create a mapping between codes, labels, checkpoints and tokenizers
checkpoint_dir = "/scratch/ka2773/project/lm-mem/checkpoints"
tokenizers = ["gpt2", "/home/ka2773/project/lm-mem/data/wikitext-103_tokenizer"]

id_to_type = {
        "a-10": ["pretrained", "gpt2", tokenizers[0]],
        "r-10": ["random", "", tokenizers[0]],
        "r-20": ["random-att", "gpt2", tokenizers[0]],
        "r-25": ["random-att-per-head", "gpt2", tokenizers[0]],
        "r-30": ["shuff-wpe", "gpt2", tokenizers[0]],
        "w-12": ["wikitext-12-layer", os.path.join(checkpoint_dir, "gpt2_40m_12-768-1024_a_", "checkpoint-26000"), tokenizers[1]],
        "w-12b": ["wikitext-12-layer", os.path.join(checkpoint_dir, "gpt2_40m_12-768-1024_a_02", "checkpoint-31000"), tokenizers[1]],
        "w-06": ["wikitext-06-layer", os.path.join(checkpoint_dir, "gpt2_40m_6-768-1024_a_", "checkpoint-18000"), tokenizers[1]],
        "w-03": ["wikitext-03-layer", os.path.join(checkpoint_dir, "gpt2_40m_3-768-1024_a_", "checkpoint-28500"), tokenizers[1]],
        "w-01": ["wikitext-01-layer", os.path.join(checkpoint_dir, "gpt2_40m_1-768-1024_a_", "checkpoint-24500"), tokenizers[1]],
        }

for model_id in ["a-10", "w-12", "w-12b"]:
    for scenario in ["sce1", "sce3", "sce4", "sce5", "sce6"]:
        for condition in ["repeat", "permute", "control"]:
            for list_type in ["random", "categorized"]:
    
                outname = "surprisal_gpt2_{}_{}_{}_{}.csv".format(model_id, 
                                                                  scenario, 
                                                                  condition,
                                                                  list_type)
                
                input_fname_path = os.path.join(root_dir, "data", "{}_lists.json".format(list_type))
                output_path = "/scratch/ka2773/project/lm-mem/output"
                python_file = os.path.join(root_dir, "gpt2_surprisal.py")
               
                # pretrained GPT-2 small
                model_type = id_to_type[model_id]
               
                # seed is only used with random/permuted models
                model_seed=12345
                
                # create command string
                command = "python {} \\\n" \
                          "--condition {} \\\n" \
                          "--scenario {} \\\n" \
                          "--paradigm with-context \\\n" \
                          "--model_type {} \\\n" \
                          "--checkpoint {} \\\n" \
                          "--path_to_tokenizer {} \\\n" \
                          "--model_seed {} \\\n" \
                          "--input_filename {} \\\n" \
                          "--output_dir {} \\\n" \
                          "--output_file {} \\\n" \
                          "--device cuda \\\n" \
                          .format(python_file,
                                  condition, 
                                  scenario,
                                  id_to_type[model_id][0],
                                  id_to_type[model_id][1],
                                  id_to_type[model_id][2],
                                  model_seed,
                                  input_fname_path, 
                                  output_path,
                                  outname)
    
                scr_filename = "script_surp_gpt2_{}_{}_{}_{}".format(model_id, 
                                                                     scenario, 
                                                                     condition,
                                                                     list_type)
                f = open(os.path.join(scripts_dir, scr_filename) + '.scr', 'w')
    
                f.write("#!/bin/bash\n")
                f.write("#SBATCH --job-name=" + scr_filename + "\n")
                f.write("#SBATCH --time=13:30:00\n")
                f.write("#SBATCH --mem 4gb\n" )
                f.write("#SBATCH --gres=gpu:1\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=1\n")
                f.write("#SBATCH --cpus-per-task=6\n")
                f.write("#SBATCH --mail-type=end\n")
                f.write("#SBATCH --mail-user=karmeni1@jhu.edu\n")
                f.write("#SBATCH --output=" + os.path.join(log_dir, scr_filename) + ".log\n")
                f.write("#SBATCH --error=" + os.path.join(log_dir, scr_filename) + ".err\n\n\n")
    
                f.write("singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif /bin/bash -c \"\n")

                f.write("source /ext3/env.sh\n")
    
                f.write("conda activate core_env\n\n")     # load environment with pytorch 1.6
                f.write(command + "\"" + "\n\n")                  # write the python command to be executed
                f.close()
    
                print("Writing {}".format(scr_filename))
                master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()
