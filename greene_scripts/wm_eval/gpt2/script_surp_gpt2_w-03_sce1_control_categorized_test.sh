python /home/ka2773/project/lm-mem/src/gpt2_surprisal.py \
--condition control \
--scenario sce1 \
--paradigm with-context \
--model_type wikitext-03-layer \
--checkpoint /scratch/ka2773/project/lm-mem/checkpoints/gpt2_40m_3-768-1024_a_/checkpoint-28500 \
--path_to_tokenizer /home/ka2773/project/lm-mem/data/wikitext-103_tokenizer \
--input_filename /home/ka2773/project/lm-mem/src/data/categorized_lists.json \
--output_dir /scratch/ka2773/project/lm-mem/output \
--output_file surprisal_gpt2_w-03_sce1_control_categorized.csv \
--device cuda \

