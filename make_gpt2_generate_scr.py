"""
script for construcing bash commands to be send as jubs to MARCC cluster
"""

master_bash = open('run_generate_scripts.sh', 'w')

for scenario in ["sce1"]:
    for strategy in ["beam", "sample"]:
        for list_type in ["random", "categorized"]:

            num_beams, do_sample = None, None
            if strategy == "beam":
                num_beams = "--num_beams 4"
                do_sample = ""
            elif strategy == "sample":
                num_beams = ""  #
                do_sample = "--do_sample "

            outname = "surprisal_{}_{}_{}.csv".format(scenario, strategy,
                                                      list_type.split(".")[0])

            list_filename = "{}_lists.json".format(list_type)

            # create command string
            command = "python generate.py --scenario {} " \
                      "--input_list ./data/{} " \
                      "--savename {} " \ 
                      "{} " \
                      "--min_tokens 10" \
                      "--max_tokens 25" \
                      "--num_beams {}" \
                      "--top_p 0.9 " \
                      "--top_k 100 " \
                      "--output_path ./output".format(scenario, list_filename, outname, do_sample, num_beams)

            scr_filename = "script_gpt2_generate_{}_{}_{}".format(scenario, strategy, list_type)
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

            f.write("conda activate ~/code/conda_envs/core_env\n\n") # load environment with pytorch 1.6
            f.write(command + "\n\n")  # write the python command to be executed
            f.close()

            master_bash.write("sbatch " + scr_filename + ".scr\n")

master_bash.close()