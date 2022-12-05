#! /bin/bash

scenario="sce1"
model_id="trf-wt103"

# bert stimuli
for list_type in "categorized" "random";
do

    for condition in "repeat" "permute" "control";
    do

        for list_len in "n3" "n5" "n7" "n10";
        do

            for prompt_key in "1" "2" "3" "4" "5";
            do

                output_name=$model_id"_"$condition"_"$scenario"_"$prompt_key"_"$list_len"_"$list_type

                python prepare_transformer_inputs.py --scenario $scenario \
                                                    --condition $condition \
                                                    --prompt_key $prompt_key \
                                                    --list_len $list_len \
                                                    --path_to_tokenizer "/home/ka2773/project/lm-mem/data/wikitext-103_tokenizer" \
                                                    --device "cuda" \
                                                    --input_filename $list_type"_lists.json" \
                                                    --output_dir "./transformer_input_files/transformer_wt103" \
                                                    --output_filename $output_name

            done
        done
    done
done