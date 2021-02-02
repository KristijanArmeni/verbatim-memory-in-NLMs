"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""

master_bash = open('run_rnn_surprisal_scripts.sh', 'w')

scenario = "sce1"

for condition in ["repeat", "permute", "control"]:
    for list_type in ["random", "categorized"]:

        outname = "surprisal_rnn_{}_{}_{}.csv".format(scenario, condition, list_type)

        nhidden=2     # number of hidden layers
        hiddendim=400 # dimensionality of hidden layer
        textgb=40     # size of input data

        model_file = "LSTM_{}_{}m_a_10-d0.2.pt".format(hiddendim, textgb)

        # create filename with markers
        test_input_file = "{}_lists_{}.txt".format(list_type, condition)
        markers_fname = test_input_file.replace(".txt", "_markers.txt")

        # create command string
        command = "python neural-complexity-master/main.py " \
                  "--model_file ./rnn_models/{} " \
                  "--vocab_file ./neural-complexity-master/vocab.txt " \
                  "--data_dir ./data " \
                  "--testfname {} " \
                  "--csvfname {} " \
                  "--markersfname {} " \
                  "--output_dir ./output " \
                  "--lowercase --test --words" \
                  .format(model_file, test_input_file, outname, markers_fname)

        scr_filename = "script_rnn_surprisal_{}_{}_{}_lists".format(scenario, condition, list_type)
        f = open(scr_filename + '.scr', 'w')

        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=" + scr_filename + "\n")
        f.write("#SBATCH --time=48:0:0\n")
        f.write("#SBATCH --partition=shared\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks-per-node=5\n")
        f.write("#SBATCH --mail-type=end\n")
        f.write("#SBATCH --mail-user=karmeni1@jhu.edu\n")
        f.write("#SBATCH --output=" + scr_filename + ".log\n")
        f.write("#SBATCH --error=" + scr_filename + ".err\n\n\n")

        f.write("ml anaconda\n")

        f.write("conda activate ~/code/conda_envs/lmpytorch1.3\n\n") # load environment with pytorch 1.3
        f.write(command + "\n\n")  # write the python command to be executed
        f.close()

        master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()

