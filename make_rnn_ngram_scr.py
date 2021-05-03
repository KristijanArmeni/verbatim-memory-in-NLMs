"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""

import os

# home directory on marcc compute cluster
# this is git versioned repo:
root_dir=os.path.expanduser("~/code/lm-mem")
scripts_dir = os.path.join(root_dir, "marcc_scripts")
log_dir = os.path.join(root_dir, 'logs')

master_bash = open(os.path.join(scripts_dir, 'run_rnn_surprisal_ngramex_scripts.sh'), 'w')

# use hyphen to separate model id strings, this makes more sense in my naming
# syntax, but replace with underscore for reading out the actual files
model_ids = ["a-10", "b-11", "c-12", "d-13", "e-14"]

for model_id in model_ids:
    for scenario in ["sce1"]:
        for condition in ["repeat"]:
            for list_type in ["ngram-random", "ngram-categorized"]:
    
                outname = "surprisal_rnn_{}_{}_{}_{}.csv".format(model_id, scenario, condition, list_type)
                
                # select a pretrained model from Van Schijndel et al (10.18653/v1/D19-1592)
                # these hyperparameters made the most significant improvement in loss
                # increasing further hasn't (see Table 1)
                nhidden=2     # number of hidden layers
                hiddendim=400 # dimensionality of hidden layer
                textgb=40     # size of input data (M tokens)
                                
                model_file = "LSTM_{}_{}m_{}-d0.2.pt".format(hiddendim, 
                                                             textgb, 
                                                             model_id.replace("-", "_"))
                model_path = os.path.join(root_dir, "rnn_models", model_file)
                
                # create absolute paths
                python_script = os.path.join(root_dir, "rnn", "main.py")
                data_dir = os.path.join(root_dir, "data")
                vocab_path = os.path.join(root_dir, "rnn", "vocab.txt")
                test_input_file = "{}_lists_{}_{}.txt".format(list_type, scenario, condition)
                markers_fname = test_input_file.replace(".txt", "_markers.txt")
                output_dir = os.path.join(root_dir, "output")
    
                # create command string
                command = "python {} " \
                          "--model_file {} " \
                          "--vocab_file {} " \
                          "--data_dir {} " \
                          "--testfname {} " \
                          "--csvfname {} " \
                          "--markersfname {} " \
                          "--output_dir {} " \
                          "--lowercase --test --words" \
                          .format(python_script,
                                  model_path,
                                  vocab_path,
                                  data_dir,
                                  test_input_file, 
                                  outname, 
                                  markers_fname,
                                  output_dir)
                
                # construct script filename, open it and write commands
                scr_filename = "script_surp_rnn_{}_{}_{}_{}".format(model_id, 
                                                                    scenario, 
                                                                    condition, 
                                                                    list_type)

                f = open(os.path.join(scripts_dir, scr_filename) + '.scr', 'w')
    
                f.write("#!/bin/bash\n")
                f.write("#SBATCH --job-name=" + scr_filename + "\n")
                f.write("#SBATCH --time=10:00:00\n")
                f.write("#SBATCH --partition=shared\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=5\n")
                f.write("#SBATCH --mail-type=end\n")
                f.write("#SBATCH --mail-user=karmeni1@jhu.edu\n")
                f.write("#SBATCH --output=" + os.path.join(log_dir, scr_filename) + ".log\n")
                f.write("#SBATCH --error=" + os.path.join(log_dir, scr_filename) + ".err\n\n\n")
    
                f.write("ml anaconda\n")
    
                f.write("conda activate ~/code/conda_envs/lmpytorch1.3\n\n") # load environment with pytorch 1.3
                f.write(command + "\n\n")  # write the python command to be executed
                f.close()
    
                print("Writing {}".format(scr_filename))
                master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()

