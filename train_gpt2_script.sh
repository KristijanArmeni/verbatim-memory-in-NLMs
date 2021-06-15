login_key=$1
wandb_name=$2
wandb_notes=$3
wandb_project="gpt2_40m"

# activate conda enviroment
ml anaconda
conda activate ~/code/conda_envs/core_env

dataset_dir=$HOME/work/karmeni1/wikitext-103

python $HOME/code/lm-mem/train_gpt2_.py --datadir $HOME/work/karmeni1/wikitext-103 \
                                        --train_ds $dataset_dir/wiki.train.inds_40m.bpe.json \
                                        --val_ds $dataset_dir/wiki.valid.inds.bpe.json \
                                        --test_ds $dataset_dir/wiki.test.inds.bpe.json \
                                        --sequence_len 12 \
                                        --model_name gpt2_40M_a.pth \
                                        --tokenizer_path $HOME/work/karmeni1/lm-mem/gpt2_wikitext103 \
                                        --seed 12345 \
                                        --device "cuda" \
                                        --train_batch_size 16 \
                                        --eval_batch_size 16 \
                                        --test_batch_size 1 \
                                        --n_layer 1 \
                                        --n_head 3 \
                                        --embed_dim 90 \
                                        --max_epochs 50 \
                                        --lr 10e3 \
                                        --betas "(0.99, 0.05)" \
                                        --num_lr_warmup_steps 5 \
                                        --es_patience 3 \
                                        --wandb_key $login_key \
                                        --wandb_dir $HOME/work/karmeni1/lm-mem/wandb \
                                        --wandb_project $wandb_project \
                                        --wandb_name $wandb_name \
                                        --savedir $HOME/work/karmeni1/lm-mem/checkpoints \
                                        --logdir $HOME/work/karmeni1/lm-mem/logs \
                                        
