#!/bin/bash

for cond in "repeat" "control" "permute";
do
    for sce in "sce1" "sce1rnd" "sce2" "sce3";
    do
        for l in "random" "categorized";
        do
            python ./data/stimuli2txt.py --json_filename "./data/"$l"_lists.json" --condition $cond --scenario_key $sce
        done
    done
done
