# Contents

This folder contains the scripts for creating the word list and stimuli.

# Data files

## Noun pools (.txt)

These are the files that contain nouns which were sampled to create noun lists

scripts: 
- `get_word_pools.py`

Downloaded noun pools are available in:

- `toronto_freq.txt` (used for sampling arbitrary lists)  
- `nouns_categorized.txt` (used for sampling semantically coherent lists)  
    
## Noun lists (.json)

scripts:
- `make_wordlists.py`

The .json files contain circularly shifted lists that are used in `surprisal_gpt2.py` and `rnn/main.py` to construct LM inputs:

- `categorized_lists.json`    
- `random_lists.json`

Each .json file contains 4 fields, each field containing 230 lists of specific length (3, 5, 7, or 10 nouns). For example:

```javascript
{ "n3": [["window", "door", "roof"], ["hall", "window", "door"], ["hearth", "hall", "window"],
```

## Intervening texts

Intervening texts for each scenario are available in `intervening_texts.json`.  
The corresponding python dicts are defined under the `prompts` varibale in `stimuli.py`  

The .json contains a dict of dicts where each level 1 dict contains texts of different lengths (1, 2, 3, 4, 5) for a specific scenario.

```python
{"sce1": {"1": "After the meeting, she took a break and had a cup of coffee. When she got back, she read the list again:", 
         "2": "After the meeting, Mary went for a walk. It was a busy day and she needed a break.       Outside was really beautiful and warm and the flowers in the park were blooming. When she got back, she read the list again:", 
         "3":
        }
}
```

Different scenario keys are:
- sce1 : intact scenario
- sce2 : incongruent scenario
- sce1rnd : scrambled scenario
- sce3 : short context scenario

For each scenario dict, the keys 1, 2, 3, 4, 5 map onto intervening text lengths (in n tokens) are as follos:
- "1": 26
- "2": 47
- "3": 99
- "4": 194
- "5": 435

## Preface strings

- `stimuli.py` contains python dicts with with preface strings (`prefixes` variable)

## LM input strings

The above materials are concatenated to create LM input strings in `make_rnn_inputfiles.py` and `gpt2_surprisal` for the LSTM and GPT-2 pipelines, respectively.

-  `make_rnn_inputfiles.py` creates *.txt and *_markers.txt files in `/rnn_input_files` and will be readin in by `rnn/main.py`  
- `make_stimuli.sh` is a script looping around `make_rnn_inputfiles.py` for creating
outputs in `/rnn_input_files`  
