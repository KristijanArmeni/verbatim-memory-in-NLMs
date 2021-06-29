# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:20:04 2021

@author: karmeni1
"""

"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_key", type=str)

args = parser.parse_args()

# home directory on marcc compute cluster
# this is git versioned repo:
root_dir = os.path.expanduser("~/code/lm-mem")
scripts_dir = os.path.join(root_dir, "marcc_scripts/train")
log_dir = os.path.join(root_dir, 'logs')

master_bash = open(os.path.join(scripts_dir, 'train_gpt2_scripts.sh'), 'w')

for model_id in ["a-10"]:
    for dataset_size in ["40M"]:
        for n_layer_n_head in [3, 6, 12]:
            for model_dim in [768]:
                
                output_path = "~/work/karmeni/lm-mem/checkpoints"
                
                python_file = os.path.join(root_dir, "traing_gpt2_.py")
                
                # toggle batch size param based on network size
                if n_layer_n_head == 3:
                    per_gpu_batch_size = 16
                else: 
                    per_gpu_batch_size = 8
                
                
                # create command string
                command = "python {}" \
                            "--train_ds $dataset_dir/wiki.train.inds_{}.bpe.json" \
                            "--val_ds $dataset_dir/wiki.valid.inds.bpe.json" \
                            "--test_ds $dataset_dir/wiki.test.inds.bpe.json" \
                            "--sequence_len 1024" \
                            "--do_train" \
                            "--model_name gpt2_40M_a.pth" \
                            "--tokenizer_path $HOME/work/karmeni1/lm-mem/gpt2_wikitext103" \
                            "--seed 12345" \
                            "--device cuda" \
                            "--train_batch_size {}" \
                            "--eval_batch_size {}" \
                            "--test_batch_size 1" \
                            "--n_layer {}" \
                            "--n_head {}" \
                            "--embed_dim {}" \
                            "--max_epochs 10" \
                            "--lr_scheduler_type cosine" \
                            "--lr 1e-5" \
                            '--betas "(0.5, 0.05)"' \
                            "--num_lr_warmup_steps 0" \
                            "--num_eval_steps 1000" \
                            "--num_logging_steps 1000" \
                            "--num_save_steps 1000 "\
                            "--es_patience 3" \
                            "--test_stride 512" \
                            "--wandb_key $login_key" \
                            "--wandb_dir $HOME/work/karmeni1/lm-mem/" \
                            "--wandb_project $wandb_project" \
                            "--wandb_group $wandb_group" \
                            "--wandb_name $wandb_name" \
                            "--wandb_notes $wandb_notes" \
                            "--wandb_mode online" \
                            "--savedir $HOME/work/karmeni1/lm-mem/checkpoints/$3" \
                            "--logdir $HOME/work/karmeni1/lm-mem/logs/$3" \
                          .format(python_file,
                                  dataset_size,
                                  per_gpu_batch_size,
                                  per_gpu_batch_size,
                                  n_layer_n_head,
                                  n_layer_n_head,
                                  model_dim)
    
                scr_filename = "train_gpt2_{}_{}_{}_{}".format(model_id,
                                                               dataset_size, 
                                                               n_layer_n_head,
                                                               model_dim)
                
                f = open(os.path.join(scripts_dir, scr_filename) + '.scr', 'w')
    
                f.write("#!/bin/bash\n")
                f.write("#SBATCH --job-name=" + scr_filename + "\n")
                f.write("#SBATCH --time=03:30:00\n")
                f.write("#SBATCH --mem=12gb\n")
                f.write("#SBATCH --partition=gpuk80\n")
                f.write("#SBATCH --gres=gpu:1\n")
                f.write("#SBATCH --cpus-per-task=6\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=1\n")
                f.write("#SBATCH --mail-type=end\n")
                f.write("#SBATCH --mail-user=karmeni1@jhu.edu\n")
                f.write("#SBATCH --output=" + os.path.join(log_dir, scr_filename) + ".log\n")
                f.write("#SBATCH --error=" + os.path.join(log_dir, scr_filename) + ".err\n\n\n")
    
                # linux input arguments
                f.write("login_key=$1")
                f.write("wandb_group=$2")
                f.write("wandb_name=$3")
                f.write("wandb_notes=$4")
                f.write('wandb_project="gpt2_{}"'.format(dataset_size))
                
                f.write("dataset_dir=$HOME/work/karmeni1/wikitext-103")
                
                f.write("ml anaconda\n")
    
                f.write("conda activate ~/code/conda_envs/core_env\n\n")    # load environment with pytorch 1.6
                f.write(command + "\n\n")                                   # write the python command to be executed
                f.close()
                
                # format arguments for the scr script
                scr_arguments = " ".join([args.wandb_key,
                                          "{}".format(dataset_size),
                                          "{}-layer".format(n_layer_n_head),
                                          "notes-here"])
    
                print("Writing {}".format(scr_filename))
                master_bash.write("sbatch " + scr_filename + ".scr " + scr_arguments + "\n")

master_bash.close()
