"""
script for construcing bash commands to be send as jobs to MARCC cluster
"""
import os

# home directory on marcc compute cluster
# this is git versioned repo:
root_dir=os.path.expanduser("~/project/lm-mem")
scripts_dir = os.path.join(root_dir, "src", "greene_scripts", "wm_eval")
log_dir = "/scratch/ka2773/project/lm-mem/logs/wm_eval"
checkpoint_root = "/scratch/ka2773/project/lm-mem/checkpoints/awd_lstm"

master_bash = open(os.path.join(scripts_dir, 'run_rnn_surprisal_scripts.sh'), 'w')

# use hyphen to separate model id strings, this makes more sense in my naming
# syntax, but replace with underscore for reading out the actual files
model_ids = ["a-10", "b-11", "c-12", "d-13", "e-14"]

modeldict = {
        
        "rnn-vs2019_400": {"dir": "vs2019",
                       "weights": "LSTM_400_40m_a_10-d0.2_statedict.pt",
                       "config_file": "LSTM_400_config.json",
                       "id": "a-10"},
        "rnn-vs2019_1600": {"dir": "vs2019",
                      "weights": "LSTM_1600_80m_a_70-d0.2_statedict.pt",
                      "config_file": "LSTM_1600_config.json",
                      "id": "a-70"},
        "rnn-gd2018": {"dir": "gd2018",
                       "weights": "hidden650_batch128_dropout0.2_lr20.0_statedict.pt",
                       "config_file": "hidden650_config.json",
                       "id": "a-10"},
        "awd-lstm-3": {"dir": "awd_lstm",
                      "weigths": "LSTM_3-layer_adam.pt",
                      "config_file": "AWD-LSTM_3-layer_config.json",
                      "id": "a-10"}
            }


for tag in modeldict.keys()[-1::]:
    for scenario in ["sce1", "sce4"]:
        for condition in ["repeat", "permute", "control"]:
            for list_type in ["random", "categorized"]:
    
                outname = "surprisal_{}_{}_{}_{}_{}.csv".format(tag.split("_")[0], 
                                                                modeldict[tag]["id"], 
                                                                scenario, 
                                                                condition, 
                                                                list_type)
                
                # create absolute paths
                python_script = os.path.join(root_dir, "src", "src", "wm_suite", "rnn", "experiment.py")
                checkpoint_folder = os.path.join(checkpoint_root, modeldict[tag]["dir"])
                model_weights = os.path.join(checkpoint_folder, modeldict[tag]["weights"]) 
                config_file = os.path.join(checkpoint_folder, modeldict[tag]["config_file"])
                vocab_file = os.path.join(checkpoint_folder, "vocab.txt")
                test_input_file = os.path.join(root_dir, "src/data/rnn_input_files", "{}_lists_{}_{}.txt".format(list_type, scenario, condition))
                markers_fname = test_input_file.replace(".txt", "_markers.txt")
                output_dir = "/scratch/ka2773/project/lm-mem/output/awd_lstm"

                # create command string
                command = "python {} " \
                          "--checkpoint_folder {} \\\n" \
                          "--model_weights {} \\\n" \
                          "--vocab_file {} \\\n" \
                          "--config_file {} \\\n" \
                          "--input_file {} \\\n" \
                          "--marker_file {} \\\n" \
                          "--scenario {} \\\n" \
                          "--condition {} \\\n" \
                          "--list_type {} \\\n" \
                          "--output_folder {} \\\n" \
                          "--output_filename {} \\\n" \
                          .format(python_script,
                                  checkpoint_folder,
                                  model_weights,
                                  vocab_file,
                                  config_file,
                                  test_input_file, 
                                  markers_fname, 
                                  scenario,
                                  condition,
                                  list_type,
                                  output_dir,
                                  outname)
                
                # construct script filename, open it and write commands
                scr_filename = "script_surp_{}_{}_{}_{}_{}".format(tag.split("_")[0],
                                                                   modeldict[tag]["id"], 
                                                                   scenario, 
                                                                   condition, 
                                                                   list_type)

                f = open(os.path.join(scripts_dir, scr_filename) + '.scr', 'w')
    
                f.write("#!/bin/bash\n")
                f.write("#SBATCH --job-name=" + scr_filename + "\n")
                f.write("#SBATCH --time=1:30:00\n")
                f.write("#SBATCH --mem=8G\n")
                f.write("#SBATCH --gres=gpu:1\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=1\n")
                f.write("#SBATCH --cpus-per-task=6\n")
                f.write("#SBATCH --mail-type=end\n")
                f.write("#SBATCH --mail-user=karmeni1@jhu.edu\n")
                f.write("#SBATCH --output=" + os.path.join(log_dir, scr_filename) + ".log\n")
                f.write("#SBATCH --error=" + os.path.join(log_dir, scr_filename) + ".err\n\n\n")
                
                f.write("singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.0-cudnn7-devel-ubuntu18.04.sif /bin/bash -c \"\n")

                f.write("source /ext3/env.sh\n")

                f.write("conda activate awd_lstm\n\n")     # load environment with pytorch 0.4

                f.write("cd /home/ka2773/project/lm-mem/src/src/wm_suite/\n")

                f.write(command + "\"" + "\n\n")                  # write the python command to be executed
                f.close()

                print("Writing {}".format(scr_filename))
                master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()

