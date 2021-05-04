#!/bin/bash

homedir=$HOME"/code/lm-mem/"

for cond in "repeat";
do
    for sce in "sce1";
    do
        for l in "ngram-random" "ngram-categorized";
        do
            python $homedir"/data/make_rnn_inputfiles.py" --json_filename $homedir"/data/"$l".json" --paradigm "repeated-ngrams" --condition $cond --scenario_key $sce
        done
    done
done
