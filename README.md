# Folder contents

- `./`         |  main scripts
- `/data`      |  scripts and .txt files for creating inputs
- `/rnn` |  rnn code by Van Schijndel et al https://github.com/vansky/neural-complexity
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

`conda env create -n env_name_gpt2 -f ./reqs_pytorch1.6.yaml`
`conda env create -n env_name_lstm -f ./reqs_pytorch1.3.yaml`

## Running [gpt2_surprisal.py](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/gpt2_surprisal.py)

Activate the installed conda enviroment with dependencies:
`conda activate env_name_gpt2`

Navigate to root folger of github repository.

Now run with the setup flag to download the models:
```bash
python gpt2_surprisal.py --setup
```

The job as follows:
```bash
python ./gpt2_surprisal.py
--condition control \
--scenario sce1 \
--paradigm with-context \
--input_filename ./data/categorized_lists.json \
--output_dir ./output \
--output_file name_of_the_output_file.csv \
--device cuda
```

Input arguments are documented in the [gpt2_surprisal.py](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/gpt2_surprisal.py) script itself:

```python
# collect input arguments
parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")

parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3"],
                    help="str, which scenario to use")
parser.add_argument("--condition", type=str, choices=["repeat", "permute", "control"],
                    help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
parser.add_argument("--paradigm", type=str, choices=["with-context", "repeated-ngrams"],
                    help="whether or not to permute the second word list")
parser.add_argument("--context_len", type=int, default=1024,
                    help="length of context window in tokens for transformers")
parser.add_argument("--model_type", type=str, default="pretrained", choices=["pretrained", "random", "random-att"],
                    help="whether or not to load a pretrained model or initialize randomly")
parser.add_argument("--model_seed", type=int, default=12345,
                    help="seed value to be used in torch.manual_seed() prior to calling GPT2Model()")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                    help="whether to run on cpu or cuda")
parser.add_argument("--input_filename", type=str,
                    help="str, the name of the .json file containing word lists")
parser.add_argument("--output_dir", type=str,
                    help="str, the name of folder to write the output_filename in")
parser.add_argument("--output_filename", type=str,
                    help="str, the name of the output file saving the dataframe")
```

## Running an LSTM job

In bash script, activate the conda environment with [LSTM dependencies](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/reqs_pytorch1.3.yaml):
`conda activate env_name_lstm`

Download LSTM model into a folder `rnn_models` from here:
https://doi.org/10.5281/zenodo.3559340 and convert them with
[rnn/model2statedict.py](./rnn/model2statedict.py) to statedicts.

You can use these commands
```bash
mkdir rnn_models
cd rnn_models
wget https://zenodo.org/record/3559340/files/LSTM_40m.tar.gz?download=1 -O LSTM_40m.tar.gz
tar -xvzf LSTM_40.tar.gz
cd ../
python rnn/model2statedict.py rnn_models/LSTM_400_40m_a_10-d0.2.pt
# or
python rnn/model2statedict.py rnn_models
```

Now you need to convert the models to statedicts.

Navigate to your root folder and use following command:

```bash
python ./rnn/experiment.py \
--checkpoint_folder checkpoints/ \
--model_weights rnn_models/LSTM_400_40m_a_10-d0.2_statedict.pt \
--vocab_file ./rnn/vocab.txt \
--config_file ./rnn/config.json \
--input_file ./data/rnn_input_files/categorized_lists_sce1_control.txt \
--marker_file ./data/rnn_input_files/categorized_lists_sce1_control_markers.txt \
--output_folder ./code/lm-mem/output \
--output_filename output_test.csv
```

## Inputs

- TODO: explain inputs
- What is sce1, sce2,...
- What is condition? repeated, permuted, control?
- What is categorized? arbitrary, categorized
