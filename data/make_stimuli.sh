#!/bin/bash

homedir=$HOME"/code/lm-mem"

for cond in "repeat" "control" "permute";
do
    for sce in "sce1" "sce1rnd" "sce2" "sce3";
    do
        for l in "random" "categorized";
        do
            python ./data/make_rnn_inputfiles.py --json_filename $homedir"/data/"$l"_lists.json" --paradigm "with-context" --condition $cond --scenario_key $sce
        done
    done
done

