# Contents

This folder contains the scripts for creating the word list and stimuli.

## Data files

### Noun lists (.json)

These files contain circularly shifted lists that are used in `surprisal_gpt2.py` and `rnn/main.py` to  
construct LM inputs.

- categorized_lists.json     
- random_lists.json

Each .json file contains 4 fields, each field containing 230 lists of specific length (3, 5, 7, or 10 nouns). For example:

```javascript
{ "n3": [["window", "door", "roof"], ["hall", "window", "door"], ["hearth", "hall", "window"],
```

### Intervening texts


## Code

- `get_word_pools.py` creates downloads these two word pools  
    - nouns_categorized.txt  
    - toronto_freq.txt  
    
- `make_wordlists.py` creates .json files containing the word lists:
    - categorized_lists.json (from nouns_categorized.txt)
    - random_lists.json (from toronto_freq.txt)

- `make_rnn_inputfiles.py` creates *.txt and *_markers.txt files in  
These files are in `/rnn_input_files` and will be readin in by `rnn/main.py`  
To be fed into the RNN.

- `make_stimuli.sh` is a script looping around `make_rnn_inputfiles.py` for creating
outputs in `/rnn_input_files`

### prefixes and prompts
- `stimuli.py` contains python dicts with prefixes and prompts

- `stimuli2txt.py` takes word lists and prefixes/prompts and writes
line-by-line into .txt files, format needed for the rnn pipeline