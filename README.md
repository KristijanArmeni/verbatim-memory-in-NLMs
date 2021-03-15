# Folder contents

- `./`         |  main scripts  
- `/data`      |  scripts and .txt files for creating inputs
- `/neural-complexity-master` |  rnn code from  https://github.com/vansky/neural-complexity
- `/output`    |  LM script outputs (.txt files)  
- `/notebooks` |  .ipynb notebooks with EDA analyses and prototyping

## Main scripts

- [experiment.py](https://github.com/KristijanArmeni/gpt2-mem/blob/main/experiment.py) contains the Experiment() class, wrapper around Transformer
library  
- [surprisal.py](https://github.com/KristijanArmeni/gpt2-mem/blob/main/surprisal.py)  script for running (calls classes/methods defined in `experiment.py`)    
- [generate.py](https://github.com/KristijanArmeni/gpt2-mem/blob/main/generate.py)  script for running text generation experiments  
- [outputs2dataframe.py](https://github.com/KristijanArmeni/gpt2-mem/blob/main/outputs4dataframe.py)  post-processing of .txt files in ./output  
