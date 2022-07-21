#! /bin/bash

scenario="sce1"
condition="repeat"
list_len="5"
model_id="bert-base-uncased"

output_name=$model_id"_"$condition"_"$scenario"_"$list_len

# bert stimuli
python prepare_transformer_inputs.py --scenario "sce1" \
                                     --condition "repeat" \
                                     --path_to_tokenizer "bert-base-uncased" \
                                     --device "cuda" \
                                     --input_filename "random_lists.json" \
                                     --output_dir "./data/transformer_input_files" \
                                     --output_filename $output_name