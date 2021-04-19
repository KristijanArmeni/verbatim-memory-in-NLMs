#!/bin/bash

datadir="~/work/karmeni1/wikitext-103"
savedir="~/work/karmeni1/lm-mem/models"

ml anaconda
conda activate ~/code/conda_envs/core_env

python ~/code/lm-mem/train_gpt2.py --train_ds $datadir/"wiki.train.tokens" \
                                   --val_ds $datadir/"wiki.valid.tokens" \
                                   --test_ds $datadir/"wiki.test.tokens" \
                                   --model_name "gpt2_40M_a.pth" \
                                   --seed 12345 \
                                   --device "cuda" \
                                   --train_batch_size 5 \
                                   --eval_batch_size 5 \
                                   --test_batch_size 1 \
                                   --max_epochs 100 \
                                   --lr 10e3 \
                                   --betas "(0.99, 0.05)" \
                                   --num_lr_warmup_steps 2000 \
                                   --es_patience 3 \
                                   --savedir $savedir \
                                   --logdir $savedir

