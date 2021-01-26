# Contents

This folder contains the scripts for creating the word list and stimuli.

## Data sources

### word lists
- `get_word_pools.py` creates downloads these two word pools  
    - nouns_categorized.txt  
    - toronto_freq.txt  
    
- `make_wordlists.py` creates .json files containing the word lists:
    - categorized_lists.json (from nouns_categorized.txt)
    - random_lists.json (from toronto_freq.txt)

### prefixes and prompts
- `stimuli.py` contains python dicts with prefixes and prompts

- `stimuli2txt.py` takes word lists and prefixes/prompts and writes
line-by-line into .txt files, format needed for the rnn pipeline