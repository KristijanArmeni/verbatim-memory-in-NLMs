# Folder contents

- `./`         |  main scripts
- `/data`      |  scripts and .txt files for creating inputs
- `/models` |  code with subfolders that contain pytorch model classes and training scripts for awd_lstm and transformer
- `/viz` |  .py script containing plotting subroutines (.ipynb notebook is not versioned)

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
  - conda-forge::wandb=0.10.31
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

Run the job as follows:
```bash
python ./wm_test_suite.py
--condition control \
--scenario sce1 \
--checkpoint gpt2
--inputs_file path_to_transformer_input_files \
--inputs_file_info path_to_transformer_input_files_info \
--context_len 1024 \
--output_dir path_to_where_outputs_are_stored \
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

## Running an AWD LSTM job

### Dependencies

If you are using conda, you can create the enviroment with `conda env create -f path_to_your_yaml_file` where yaml file can look like:

```yaml
name: awd_lstm
channels:
  - defaults
  - pytorch
dependencies:
  - python
  - ipython
  - pytorch=1.0.1
  - cudatoolkit=10.0
  - cupy
  - numpy
  - scipy
  - pandas
  - nltk
  - pip
  - pip:
    - pynvrtc
```

Navigate to your root folder and use the following commands:

```bash
conda activate awd_lstm  # activate the conda environment with dependencies

cd /home/ka2773/project/lm-mem/src/src/wm_suite/awd_lstm  # make sure we're in the awd_lstm folder for module imports

python ./wm_suite/experiment.py \
--checkpoint_folder path/to/folder/with/awd_lstm/checkpoint/folder \
--model_weights path/to/folder/with/awd_lstm/checkpoints/weights.pt \
--vocab_file awd_lstm_corpus \
--config_file /scratch/ka2773/project/lm-mem/checkpoints/awd_lstm/AWD-LSTM_3-layer_config.json \
--input_file /home/ka2773/project/lm-mem/src/data/rnn_input_files/random_lists_sce1_permute.txt \
--marker_file /home/ka2773/project/lm-mem/src/data/rnn_input_files/random_lists_sce1_permute_markers.txt \
--per_token_output \
--output_folder /scratch/ka2773/project/lm-mem/output/awd_lstm \
--output_filename surprisal_awd-lstm-3_a-10_sce1_permute_random.csv \
"
```
