#!/bin/bash

for cond in "repeat";
do
    for sce in "sce1";
    do
        for l in "ngram-random";
        do
            python ./data/make_rnn_stimuli.py --json_filename "./data/"$l"_lists.json" --paradigm "ngram-random" --condition $cond --scenario_key $sce
        done
    done
done
