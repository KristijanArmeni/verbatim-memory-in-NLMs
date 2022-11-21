# Characterizing Short-term Memory In Neural Language Models

![thumbnail](thumbnail_2x.png)

ArXiV preprint:  https://arxiv.org/abs/2210.13569

# Folder contents

- `./`         |  main scripts
- `/data`      |  scripts and .txt files for creating inputs
- `/models` |  code with subfolders that contain pytorch model classes and training scripts for awd_lstm and transformer.
- `/wm_suite` | code with script for evaluating and analysing models on the working memory test suite
- `/viz` |  .py script containing plotting subroutines (.ipynb notebook is not versioned)

# Dependencies

Dependencies are specified in the [main_env.yml](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/awd-lstm_env.yml) and [awd-lstm_env.yml](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/awd-lstm_env.yml) files for the transformer and awd-lstm code-bases, respectively.

We used the conda management toolkit, so the easiest way to create the environment with dependencies is as follows:

`conda env create -n your_env_name -f ./main_env.yml`



```
