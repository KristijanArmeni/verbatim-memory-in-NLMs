#! /bin/bash

sourcedir="/home/ka2773/project/lm-mem/src/data/wt103"

python $sourcedir/dataset.py --train_tokens $HOME/project/lm-mem/data/wikitext-103/wiki.train.tokens \
                             --valid_tokens $HOME/project/lm-mem/data/wikitext-103/wiki.valid.tokens \
                             --test_tokens $HOME/project/lm-mem/data/wikitext-103/wiki.test.tokens \
                             --train_tokenizer \
                             --tokenizer_train_tokens $HOME/project/lm-mem/data/wikitext-103/wiki.train.tokens \
                             --train_savename wiki.train.tokens.bpe \
                             --valid_savename wiki.valid.tokens.bpe \
                             --test_savename wiki.test.tokens.bpe \
                             --tokenizer_savedir $HOME/project/lm-mem/data/wikitext-103_v2/tokenizer \
                             --savedir $HOME/project/lm-mem/data/wikitext-103_v2
