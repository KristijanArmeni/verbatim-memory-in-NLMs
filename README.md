# Folder contents

- `./`         |  main scripts
- `/data`      |  scripts and .txt files for creating inputs
- `/rnn` |  rnn code by Van Schijndel et al https://github.com/vansky/neural-complexity
- `/output`    |  LM script outputs (.txt files)
- `/notebooks` |  .ipynb notebooks with EDA analyses and prototyping

# Dependencies

For our experiments, we used the following dependencies (environment.yml):

channels:
  - defaults
dependencies:
  - numpy
  - huggingface::transformers=4.6
  - nb_conda_kernels
  - jupyterlab
  - matplotlib
  - h5py
  - seaborn
  - pytorch==1.10.1
  - scikit-learn
  - scipy
  - nltk=3.4.4
  - pip
  - python=3.7
  - cudatoolkit=11.3
  - plotly
  - conda-forge::wandb=0.10.31
  - mne=0.22.0
  - pytorch-lightning=1.5.10
  - gensim=3.8

## Installing dependencies

The dependencies are specified in the .yml file: [environement.yml](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/enviroment.yml)


We used the conda management toolkit, so the easiest way to create the environment with dependencies is as follows:

`conda env create -n your_env_name -f ./environment.yml`

## Running [wm_test_suite.py](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/gpt2_surprisal.py)

Activate the installed conda enviroment with dependencies:  
```
conda activate your_env_name
```

Navigate to root folder of github repository.

Now run with the setup flag to download the models:
```bash
python wm_test_suite.py --setup
```

Run the job as follows:
```bash
python ./wm_test_suite.py
--condition control \
--scenario sce1 \
--paradigm with-context \
--input_filename ./data/categorized_lists.json \
--output_dir ./output \
--output_file name_of_the_output_file.csv \
--device cuda
```

Input arguments are documented in the [wm_test_suite.py](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/wm_test_suite.py) script itself:

```python
parser.add_argument("--setup", action="store_true",
                    help="downloads and places nltk model and Tokenizer")
parser.add_argument("--scenario", type=str, choices=["sce1", "sce1rnd", "sce2", "sce3", "sce4", "sce5", "sce6", "sce7"],
                    help="str, which scenario to use")
parser.add_argument("--condition", type=str)
parser.add_argument("--inputs_file", type=str, help="json file with input sequence IDs which are converted to tensors")
parser.add_argument("--inputs_file_info", type=str, help="json file with information about input sequences")
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--context_len", type=int, default=1024,
                    help="length of context window in tokens for transformers")
parser.add_argument("--model_type", type=str,
                    help="model label controlling which checkpoint to load")
# To download a different model look at https://huggingface.co/models?filter=gpt2
parser.add_argument("--checkpoint", type=str, default="gpt2",
                    help="the path to folder with pretrained models (expected to work with model.from_pretraiend() method)")
parser.add_argument("--model_seed", type=int, default=12345,
                    help="seed value to be used in torch.manual_seed() prior to calling GPT2Model()")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                    help="whether to run on cpu or cuda")
parser.add_argument("--output_dir", type=str,
                    help="str, the name of folder to write the output_filename in")
parser.add_argument("--output_filename", type=str,
                    help="str, the name of the output file saving the dataframe")
```

## Running an LSTM job

In bash script, activate the conda environment with [dependencies]()
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
