#!/bin/bash

python /home/ka2773/project/lm-mem/src/rnn_/experiment.py --checkpoint_folder /scratch/ka2773/project/lm-mem/lstm/checkpoints/vs2019 --model_weights /scratch/ka2773/project/lm-mem/checkpoints/lstm/vs2019/LSTM_1600_80m_a_70-d0.2_statedict.pt \
--vocab_file /scratch/ka2773/project/lm-mem/checkpoints/lstm/vs2019/vocab.txt \
--config_file /scratch/ka2773/project/lm-mem/checkpoints/lstm/vs2019/LSTM_1600_config.json \
--input_file /home/ka2773/project/lm-mem/src/data/rnn_input_files/categorized_lists_sce1_control.txt \
--marker_file /home/ka2773/project/lm-mem/src/data/rnn_input_files/categorized_lists_sce1_control_markers.txt \
--scenario sce1 \
--condition control \
--list_type categorized \
--output_folder /scratch/ka2773/project/lm-mem/output \
--output_filename surprisal_rnn-vs1600_sce1_categorized_control.csv \

