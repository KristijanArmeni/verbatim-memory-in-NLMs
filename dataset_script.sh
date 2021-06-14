#!/bin/bash

ml anaconda
conda activate ~/code/conda_envs/core_env

sourcedir=$HOME/code/lm-mem
datadir=$HOME/work/karmeni1/wikitext-103

python $sourcedir/dataset.py --train_tokens $datadir/wiki.train.tokens_40m \
                             --valid_tokens $datadir/wiki.valid.tokens \
                             --test_tokens $datadir/wiki.test.tokens \
                             --train_tokenizer \
                             --tokenizer_train_tokens $datadir/wiki.train.tokens \
                             --tokenizer_savedir $HOME/work/karmeni1/lm-mem/gpt2_wikitext103 \
                             --saveidr $datadir
