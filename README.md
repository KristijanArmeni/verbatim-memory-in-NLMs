# Folder contents

- `./`         |  main scripts  
- `/data`      |  scripts and .txt files for creating inputs
- `/rnn` |  rnn code from  https://github.com/vansky/neural-complexity
- `/output`    |  LM script outputs (.txt files)  
- `/notebooks` |  .ipynb notebooks with EDA analyses and prototyping

# Dependencies

For our experiments, we used the following dependencies:

  - python=3.7
  - matplotlib
  - scipy
  - numpy
  - nltk=3.4.4
  - plotly
  - seaborn
  - pytorch::pytorch=1.6=*cuda9.2*
  - huggingface::transformers=4.2.2
  - nb_conda_kernels
  - h5py
  - pip

## Installing dependencies

There are two separate sets of dependencies, defined in the requirements files: 
- [reqs_pytorch1.3.yaml](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/reqs_pytorch1.3.yaml) is for the rnn code
- [reqs_pytorch1.6.yaml](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/reqs_pytorch1.6.yaml) is for the gpt-2 code.

Pytorch 1.3 was used with RNN code base because the model objects that stored checkpoints contained attributes that were not compatible with pytorch 1.6.

We used the conda management toolkit, so the easiest way to create the environment with dependencies is as follows:  

`conda create /path/to/where/dependencies/are/installed -f ./reqs_pytorch1.3.yaml`

## Main scripts

- [gpt2_surprisal.py](https://github.com/KristijanArmeni/gpt2-mem/blob/main/surprisal.py)  script for running (calls classes/methods defined in `experiment.py`)    
