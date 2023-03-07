#! /bin/bash

scenario="sce1"
model_id="gpt2"

# bert stimuli
for list_type in "random";
do

    for condition in "repeat";
    do

        for list_len in "n5";
        do

            for prompt_key in "1";
            do

                output_name=$model_id"_"$condition"_"$scenario"_"$prompt_key"_"$list_len"_"$list_type

                python prepare_transformer_inputs.py --scenario $scenario \
                                                    --condition $condition \
                                                    --prompt_key $prompt_key \
                                                    --list_len $list_len \
                                                    --pretokenize_moses \
                                                    --path_to_tokenizer "/home/ka2773/project/lm-mem/data/wikitext-103_v2/tokenizer" \
                                                    --device "cuda" \
                                                    --input_filename $list_type"_lists.json" \
                                                    --output_dir "/home/ka2773/project/lm-mem/src/data/transformer_input_files/transformer_wt103_v2" \
                                                    --output_filename $output_name

            done
        done
    done
done
