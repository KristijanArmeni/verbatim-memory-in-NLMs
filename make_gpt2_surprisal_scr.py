"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""

master_bash = open('run_gpt2_surprisal_scripts.sh', 'w')

for scenario in ["sce1", "sce1rnd"]:
    for condition in ["repeat", "permute", "control"]:
        for list_type in ["random", "categorized"]:

            outname = "surprisal_gpt2_{}_{}_{}.csv".format(scenario, condition,
                                                      list_type)

            # create command string
            command = "python surprisal.py --condition {} --scenario {} " \
                      "--input_filename ./data/{} " \
                      "--output_dir ./output --output_file {}"\
                      .format(condition, scenario, "{}_lists.json".format(list_type), outname)

            scr_filename = "script_gpt2_surprisal_{}_{}_{}".format(scenario, condition,
                                                            list_type)
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

            f.write("conda activate ~/code/conda_envs/core_env\n\n")    # load environment with pytorch 1.6
            f.write(command + "\n\n")                                   # write the python command to be executed
            f.close()

            print("Writing {}".format(scr_filename))
            master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()