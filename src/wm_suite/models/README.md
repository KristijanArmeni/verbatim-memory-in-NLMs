This folder contains the scripts and modules for training the [AWD LSTM](#AWD_LSTM) and the training script for the [Wikitext-103 transformer model](#Wikitext-103-Transformer).


# AWD_LSTM

This is a copy of the [original Salesforce AWD_LSTM github repository](https://github.com/salesforce/awd-lstm-lm).

Note that training the awd lstm required a different set of dependencies (e.g. an earlier pytorch version) than training the transformer model.

The awd lstm dependencies are specified in the [awd-lstm_env.yml](https://github.com/KristijanArmeni/neural-lm-mem/blob/main/awd-lstm_env.yml) file.

An example hpc job script looked like this:

```
# activate conda environment with dependencies
conda activate awd_lstm

# let's go to the awd_lstm root folder
cd path/to/folder/with/awd_lstm/code

# run the script
python -u main.py --epochs 50 \
                  --nlayers 3 \
                  --emsize 400 \
                  --nhid 1840 \
                  --alpha 0 \
                  --beta 0 \
                  --dropoute 0 \
                  --dropouth 0.01 \
                  --dropouti 0.01 \
                  --dropout 0.4 \
                  --wdrop 0.2 \
                  --wdecay 1.2e-6 \
                  --bptt 200 \
                  --batch_size 128 \
                  --optimizer adam \
                  --lr 1e-3 \
                  --data data/wikitext-103 \
                  --save LSTM_3-layer_adam.pt \
                  --when 25 35 \
                  --model LSTM
```

The hyperparameters used for training are also described in the Appendix of the [preprint](https://arxiv.org/abs/2210.13569).

# Wikitext-103 Transformer

The dependencies we used are specified in the main_env.yml file.

You can download the model checkpoint [from the Hugging Face Hub](https://huggingface.co/Kristijan/gpt2_wt103-40m_12-layer).

An example training job would like something like this:

```bash
# activate environment with pytorch1.6
conda activate core_env

# call training script
python $HOME/project/lm-mem/src/train_gpt2_.py \ 
                           --train_ds $dataset_dir/wiki.train.inds_$ds_size.bpe.json \
                           --val_ds $dataset_dir/wiki.valid.inds.bpe.json \
                           --test_ds $dataset_dir/wiki.test.inds.bpe.json \
                           --sequence_len 1024 \
                           --do_train \
                           --model_name $model_filename \
                           --tokenizer_path $HOME/project/lm-mem/data/wikitext-103_tokenizer \
                           --seed $seed \
					       --device cuda \
					       --train_batch_size $train_batch_size \
					       --eval_batch_size $eval_batch_size \
					       --test_batch_size $test_batch_size \
					       --n_layer $n_layer \
					       --n_head $n_head \
					       --embed_dim $n_embed \
					       --max_epochs $max_epochs \
					       --lr_scheduler_type $lr_scheduler_type \
					       --lr $lr \
					       --adam_beta1 $adam_beta1 \
					       --adam_beta2 $adam_beta2 \
					       --num_lr_warmup_steps 0 \
					       --num_eval_steps 500 \
					       --num_logging_steps 500 \
					       --num_save_steps 500 \
					       --es_patience $es_patience \
					       --es_delta $es_delta \
					       --test_stride $test_stride \
					       --wandb_key $login_key \
					       --wandb_dir $SCRATCH/project/lm-mem/ \
					       --wandb_project $wandb_project \
					       --wandb_group $wandb_group \
					       --wandb_name $wandb_name \
					       --wandb_notes $wandb_notes \
					       --wandb_mode online \
					       --savedir $SCRATCH/project/lm-mem/checkpoints/$model_name \
					       --logdir $SCRATCH/project/lm-mem/logs/$model_name
```

The variables that set the hyperparameter values can be sourced from a `.config` file (e.g. in bash: `source config_file_with_hyperparams.config`) whic can look like this:

```bash
ds_size="40m"
seq_len=1024
n_embed=768
n_layer=12
n_head=12
seed=12345
train_batch_size=12
eval_batch_size=12
test_batch_size=1
test_stride=256
max_epochs=10
lr_scheduler_type="constant"
lr=6e-5
adam_beta1=0.6
adam_beta2=0.1
es_patience=5
es_delta=0.01
```

Most of these arguments are just passed on to the `.train()` method of the HugginFace [Trainer()](https://huggingface.co/transformers/v4.6.0/main_classes/trainer.html) class. Some are arguments for Weights and Biases experiment tracking and some are just paths to local folders.
