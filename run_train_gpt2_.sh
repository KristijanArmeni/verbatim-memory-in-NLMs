login_key=$1

# activate conda enviroment
ml anaconda
conda activate ~/code/conda_envs/core_env

python $HOME/code/lm-mem/train_gpt2_.py --datadir "C:\Users\karmeni1\project\lm-mem\data\wikitext-103\" \
                                        --train_ds "wiki.train.tokens" \
                                        --val_ds "wiki.valid.tokens" \
                                        --test_ds  "wiki.test.tokens" \
                                        --sequence_len 12 \
                                        --model_name "gpt2_40M_a.pth" \
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
                                        --savedir "C:\Users\karmeni1\project\lm-mem\data\checkpoints\" \
                                        --logdir "C:\Users\karmeni1\project\lm-mem\data\logs\
                                        
