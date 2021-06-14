#!/bin/bash

conda activate ~/code/conda_envs/core_env

wikipath=$HOME/work/karmeni1/wikitext-103

# here we train tokenizer on Wikitext-103 and retokenize the dataset
python dataset.py --train_tokens $wikipath/wiki.train.tokens_40m \
                  --valid_tokens $wikipath/wiki.valid.tokens \
                  --test_tokens $wikipath/wiki.test.tokens \
                  --tokenizer_train_tokens $wikipath/wiki.train.tokens \
                  --train_tokenizer True \
                  --tokenizer_savedir $HOME/work/karmeni1/lm-mem/gpt2_wikitext103 \
                  --savedir $wikipath
                  
