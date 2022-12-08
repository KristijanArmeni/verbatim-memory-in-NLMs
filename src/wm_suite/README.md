
# About

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
