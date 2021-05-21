"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""

import os

# home directory on marcc compute cluster
# this is git versioned repo:
root_dir=os.path.expanduser("~/code/lm-mem")
scripts_dir = os.path.join(root_dir, "marcc_scripts")
log_dir = os.path.join(root_dir, 'logs')

master_bash = open(os.path.join(scripts_dir, 'run_gpt2_surprisal_ngramex_scripts.sh'), 'w')

for model_id in ["a-10", "r-10"]:
    for scenario in ["sce1"]:
        for condition in ["repeat"]:
            for list_type in ["ngram-random", "ngram-categorized"]:
    
                outname = "surprisal_gpt2_{}_{}_{}_{}.csv".format(model_id,
                                                                  scenario, 
                                                                  condition,
                                                                  list_type)
                
                input_fname_path = os.path.join(root_dir, "data", "{}.json".format(list_type))
                output_path = os.path.join(root_dir, "output")
                python_file = os.path.join(root_dir, "gpt2_surprisal.py")
                
                # pretrained GPT-2 small model
                model_type = "pretrained"
                model_seed = None
                
                # randomly initialized model
                if model_id == "r-10":
                    model_type = "random"
                    model_seed = 12345
                
                # model with Q, and K (attention) matrices permuted
                elif model_id == "r-20":
                    model_type = "random-att"
                    model_seed = 12345
                
                # create command string
                command = "python {} --condition {} --scenario {} " \
                          "--paradigm repeated-ngrams " \
                          "--model_type {} " \
                          "--model_seed {} " \
                          "--input_filename {} " \
                          "--output_dir {} --output_file {} " \
                          "--device cuda"\
                          .format(python_file,
                                  condition, scenario,
                                  model_type,
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
                f.write("#SBATCH --time=03:30:00\n")
                f.write("#SBATCH --mem=4gb\n")
                f.write("#SBATCH --partition=gpuk80\n")
                f.write("#SBATCH --gres=gpu:1\n")
                f.write("#SBATCH --cpus-per-task=6\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=1\n")
                f.write("#SBATCH --mail-type=end\n")
                f.write("#SBATCH --mail-user=karmeni1@jhu.edu\n")
                f.write("#SBATCH --output=" + os.path.join(log_dir, scr_filename) + ".log\n")
                f.write("#SBATCH --error=" + os.path.join(log_dir, scr_filename) + ".err\n\n\n")
    
                f.write("ml anaconda\n")
    
                f.write("conda activate ~/code/conda_envs/core_env\n\n")    # load environment with pytorch 1.6
                f.write(command + "\n\n")                                   # write the python command to be executed
                f.close()
    
                print("Writing {}".format(scr_filename))
                master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()
