login_key=$1

# activate conda enviroment
ml anaconda
conda activate ~/code/conda_envs/core_env

python $HOME/code/lm-mem/train_gpt2_.py --datadir $HOME/work/karmeni1/wikitext-103 \
                                        --train_ds wiki.train.inds_40m.bpe.json \
                                        --val_ds wiki.valid.inds.bpe.json \
                                        --test_ds wiki.test.inds.bpe.json \
                                        --sequence_len 12 \
                                        --model_name gpt2_40M_a.pth \
                                        --seed 12345 \
                                        --device "cuda" \
                                        --train_batch_size 10 \
                                        --eval_batch_size 10 \
                                        --test_batch_size 1 \
                                        --n_layer 1 \
                                        --n_head 3 \
                                        --max_epochs 100 \
                                        --lr 10e3 \
                                        --betas "(0.99, 0.05)" \
                                        --num_lr_warmup_steps 5 \
                                        --es_patience 3 \
                                        --wandb_key $login_key \
                                        --savedir $HOME/work/karmeni1/lm-mem/checkpoints \
                                        --logdir $HOME/work/karmeni1/lm-mem/logs \
                                        
