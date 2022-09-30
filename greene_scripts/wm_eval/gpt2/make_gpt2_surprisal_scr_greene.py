"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""
import os
import json

# home directory on marcc compute cluster
# this is git versioned repo:
root_dir = os.path.expanduser("~/project/lm-mem/src")
scripts_dir = os.path.join(root_dir, "greene_scripts", "wm_eval", "bert")
log_dir = "/scratch/ka2773/project/lm-mem/logs/wm_eval/bert"

# open the main file that will submit the jobs
master_bash = open(os.path.join(scripts_dir, 'run_bert_surprisal_scripts.sh'), 'w')

# load .json dict with paths to best checkpoints
checkpoint_config_file = os.path.join(root_dir, "greene_scripts", "wm_eval", "checkpoint_configs.json")
with open(checkpoint_config_file, 'r') as f:
    id_to_type = json.load(f)

# now create the scripts with all the input args etc.
for model_id in ["b-10"]:
    for scenario in ["sce1"]:
        for list_len in ["n3", "n5", "n7", "n10"]:
            for prompt_len in ["1", "2", "3", "4", "5"]:
                for condition in ["repeat", "permute", "control"]:
                    for list_type in ["categorized"]:
    
                        outname = "surprisal_bert_{}_{}_{}_{}_{}_{}.csv".format(model_id, 
                                                                                scenario,
                                                                                prompt_len,
                                                                                list_len,
                                                                                condition,
                                                                                list_type)
                        
                        inputs_file = os.path.join(root_dir, 
                                                   "data", 
                                                   "transformer_input_files", 
                                                   "{}_{}_{}_{}_{}_{}.json".format(id_to_type[model_id][-1],
                                                                                   condition,
                                                                                   scenario,
                                                                                   prompt_len,
                                                                                   list_len,
                                                                                   list_type))
                        
                        inputs_file_info = inputs_file.replace(".json", "_info.json")
                        
                        output_path = "/scratch/ka2773/project/lm-mem/output/bert"
                        python_file = os.path.join(root_dir, "wm_test_suite.py")
                    
                        # pretrained GPT-2 small
                        model_type = id_to_type[model_id]
                    
                        # seed is only used with random/permuted models
                        model_seed=12345
                        
                        # create command string
                        command = "python {} \\\n" \
                                "--condition {} \\\n" \
                                "--scenario {} \\\n" \
                                "--model_type {} \\\n" \
                                "--checkpoint {} \\\n" \
                                "--tokenizer {} \\\n" \
                                "--model_seed {} \\\n" \
                                "--inputs_file {} \\\n" \
                                "--inputs_file_info {} \\\n" \
                                "--context_len {} \\\n" \
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
                                        inputs_file,
                                        inputs_file_info, 
                                        512,
                                        output_path,
                                        outname)
            
                        scr_filename = "script_surp_bert_{}_{}_{}_{}_{}_{}".format(model_id, 
                                                                                       scenario, 
                                                                                       list_len,
                                                                                       prompt_len,
                                                                                       condition,
                                                                                       list_type)

                        f = open(os.path.join(scripts_dir, scr_filename) + '.scr', 'w')
            
                        f.write("#!/bin/bash\n")
                        f.write("#SBATCH --job-name=" + scr_filename + "\n")
                        f.write("#SBATCH --time=01:30:00\n")
                        f.write("#SBATCH --mem 4gb\n" )
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
