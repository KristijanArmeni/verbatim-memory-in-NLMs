"""
surprisal.py is used to run the perplexity experiment with GPT-2
it relies on the Sampler() class which is just a class with wrapper methods
around the Transformers library.

Use as:

python experiment.py
or from an ipython console:
%run experiment.py ""

"""

import numpy as np
import pandas as pd
import json
import sys, os
import argparse

# own modules
sys.path.append(os.path.join(os.getcwd(), 'data'))
from experiment import model, tokenizer, Sampler, run_perplexity
from stimuli import prefixes, prompts


# ===== INITIATIONS ===== #
# collect input arguments
parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")

parser.add_argument("--scenario", type=str, choices=["sce1", "sce2"],
                    help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
parser.add_argument("--condition", type=str,
                    help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
parser.add_argument("--input_filename", type=str,
                    help="str, the name of the .json file containing word lists")
parser.add_argument("--output_dir", type=str,
                    help="str, the name of folder to write the output_filename in")
parser.add_argument("--output_filename", type=str,
                    help="str, the name of the output file saving the dataframe")

argins = parser.parse_args()

#condition = sys.argv[1]   # can be one of: "repeat", "permute"
#input_filename = sys.argv[2]    # the .json file to be read in
#outname = sys.argv[3]  # the name of the output data frame

# uncomment this for debugging
#condition = "permute"
#fname = "./data/semantic_lists.json"
#outname = "perplexity_semantic.csv"

# construct output file name and check that it exists
savedir = os.path.join(".", argins.output_dir)
assert os.path.isdir(savedir)               # check that the folder exists
base, extension = os.path.splitext(argins.output_filename)
outpath = os.path.join(savedir, "".join([base, "_", argins.condition, extension]))

print("condition == {}".format(argins.condition))
print("scenario == {}".format(argins.condition))

# initiate the sampler class from experiment.py module
s = Sampler(model=model, tokenizer=tokenizer)

# ===== DATASET ===== #
# load the word lists in .json files
with open(argins.input_filename) as f:
    print("Loading {} ...".format(argins.input_filename))
    stim = json.load(f)

# convert word lists to strings and permute the second one if needed
# add space at the string onset
word_list1 = [" " + ", ".join(l) + "." for l in stim]
if argins.condition == "permute":
    word_list2 = [" " + ", ".join(np.random.RandomState((543+j)*5).permutation(stim[j]).tolist()) + "."
                  for j in range(len(stim))]
else:
    word_list2 = word_list1

# ===== COMPUTE PERPLEXITY ===== #

# call the wrapper function
# this one loops over prefixes and over prompts
output_list = run_perplexity(prefixes=prefixes,
                             prompts={key: prompts[argins.scenario][key] for key in ["7"]},
                             word_list1=word_list1,
                             word_list2=word_list2,
                             sampler=s)

# ===== FORMAT AND SAVE OUTPUT ===== #

# convert the output to dataframe
dfout = []
counter = 1  # counter for trials

# loop over trials
for k, tup in enumerate(output_list):

    # convert the last two elements of the tuple to an array
    dftmp = pd.DataFrame(np.asarray(tup[1:5]).T,
                         columns=["ppl", "token", "trialID", "positionID"])

    dftmp["ispunct"] = dftmp.token.isin([".", ":", ","])  # create punctuation info column
    dftmp['prefix'] = tup[-2]                             # add a column of prefix labels
    dftmp['prompt'] = tup[-1]                             # add a column of prompt labels
    dftmp['stimID'] = counter
    dftmp['second_list'] = argins.condition                      # log condition of the second list

    dfout.append(dftmp)
    counter += 1

# put into a single df and save
dfout = pd.concat(dfout)

# save output
print("Saving {}".format(outpath))
dfout.to_csv(outpath, sep=",")