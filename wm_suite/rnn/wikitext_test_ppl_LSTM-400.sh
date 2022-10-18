#!/bin/bash

python /home/ka2773/project/lm-mem/src/rnn/experiment.py \
--checkpoint_folder /scratch/ka2773/project/lm-mem/checkpoints/lstm/vs2019 \
--model_weights /scratch/ka2773/project/lm-mem/checkpoints/lstm/vs2019/LSTM_400_40m_a_10-d0.2_statedict.pt \
--vocab_file /scratch/ka2773/project/lm-mem/checkpoints/lstm/vs2019/vocab.txt \
--config_file /scratch/ka2773/project/lm-mem/checkpoints/lstm/vs2019/LSTM_400_config.json \
--input_file /home/ka2773/project/lm-mem/data/wikitext-103/wiki.test.tokens.nolinebreaks \
--marker_file '' \
--output_folder /scratch/ka2773/project/lm-mem/output \
--output_filename LSTM_400_40m_a_10-d0.2_test_ppl.csv \

