login_key=$1
wandb_group=$2
wandb_name=$3
wandb_notes=$4
wandb_project="gpt2_40m"

# activate conda enviroment
ml anaconda
conda activate ~/code/conda_envs/core_env

dataset_dir=$HOME/work/karmeni1/wikitext-103

python $HOME/code/lm-mem/train_gpt2_.py --datadir $HOME/work/karmeni1/wikitext-103 \
                                                --train_ds $dataset_dir/wiki.train.inds_40m.bpe.json \
                                                --val_ds $dataset_dir/wiki.valid.inds.bpe.json \
                                                --test_ds $dataset_dir/wiki.test.inds.bpe.json \
                                                --sequence_len 1024 \
                                                --do_train \
                                                --model_name gpt2_40M_a.pth \
                                                --tokenizer_path $HOME/work/karmeni1/lm-mem/gpt2_wikitext103 \
                                                --seed 12345 \
                                                --device "cuda" \
                                                --train_batch_size 12 \
                                                --eval_batch_size 12 \
                                                --test_batch_size 1 \
                                                --n_layer 2 \
                                                --n_head 2 \
                                                --embed_dim 252 \
                                                --max_epochs 10 \
                                                --lr_scheduler_type "constant" \
                                                --lr 1e-5 \
                                                --betas "(0.5, 0.05)" \
                                                --num_lr_warmup_steps 0 \
                                                --num_eval_steps 1000 \
                                                --num_logging_steps 1000 \
                                                --num_save_steps 1000 \
                                                --es_patience 3 \
                                                --test_stride 512 \
                                                --wandb_key $login_key \
                                                --wandb_dir $HOME/work/karmeni1/lm-mem/ \
                                                --wandb_project $wandb_project \
                                                --wandb_group "$wandb_group" \
                                                --wandb_name "$wandb_name" \
                                                --wandb_notes "$wandb_notes" \
                                                --wandb_mode "online" \
                                                --savedir $HOME/work/karmeni1/lm-mem/checkpoints/$3 \
                                                --logdir $HOME/work/karmeni1/lm-mem/logs/$3 
                                        
