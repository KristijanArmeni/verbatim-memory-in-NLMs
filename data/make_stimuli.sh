#!/bin/bash

homedir=$HOME"/project/lm-mem/src"

for cond in "repeat" "control" "permute";
do
    for sce in "sce4" "sce5" "sce6";
    do
        for l in "random" "categorized";
        do
            python $homedir"/data/make_rnn_inputfiles.py" --json_filename $homedir"/data/"$l"_lists.json" --paradigm "with-context" --condition $cond --scenario_key $sce
        done
    done
done

