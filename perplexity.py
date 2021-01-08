"""
perplexity.py is used to run the perplexity experiment with GPT-2
it relies on the Sampler() class which is just a class with wrapper methods
around the Transformers library.

Use as:

python experiment.py
or from an ipython console:
%run experiment.py

"""

import numpy as np
import pandas as pd
import json
import sys

# own modules
from experiment import model, tokenizer, Sampler, run_perplexity
from stimuli import prefixes_perplexity, prompts_perplexity

# ===== INITIATE ===== #
permute_second_list = bool(int(sys.argv[1]))
print("permute_second_list == {}".format(permute_second_list))

s = Sampler(model=model, tokenizer=tokenizer)

# ===== DATASET ===== #
# load the word lists
fname = "./data/toronto.json"

with open(fname) as f:
    print("Loading {} ...".format(fname))
    stim = json.load(f)

# convert word lists to strings and permute the second one if needed
word_list1 = [", ".join(l) + "." for l in stim]
if permute_second_list:
    word_list2 = [", ".join(np.random.RandomState((543+j)*5).permutation(stim[j]).tolist()) + "."
                  for j in range(len(stim))]
else:
    word_list2 = word_list1

# ===== COMPUTE PERPLEXITY ===== #

# call the wrapper function
output_list = run_perplexity(prefixes=prefixes_perplexity,
                             prompts={key: prompts_perplexity[key] for key in ["sce1-1", "sce1-3", "sce1-5"]},
                             word_list1=word_list1,
                             word_list2=word_list2,
                             sampler=s)

# convert the output to dataframe
dfout = []
counter = 1
for k, tup in enumerate(output_list):
    # convert the last two elements of the tuple to an array
    dftmp = pd.DataFrame(np.asarray(tup[1:5]).T, columns=["ppl", "token", "trialID", "positionID"])
    dftmp["ispunct"] = dftmp.token.isin([".", ":", ","])  # create punctuation info column
    dftmp['prefix'] = tup[-2]                             # add a column of prefix labels
    dftmp['prompt'] = tup[-1]                             # add a column of prompt labels
    dftmp['stimID'] = counter

    dfout.append(dftmp)
    counter += 1

# put into df and save
dfout = pd.concat(dfout)

# save output
outname = None
if permute_second_list:
    outname = "./perplexity3B.txt"
else:
    outname = "./perplexity3.txt"

print("Saving {}".format(outname))
dfout.to_csv(outname, sep=",")
