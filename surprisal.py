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
from experiment import model, tokenizer, Sampler
from stimuli import prefixes, prompts
import torch

# ===== INITIATIONS ===== #
# collect input arguments
parser = argparse.ArgumentParser(description="surprisal.py runs perplexity experiment")

parser.add_argument("--scenario", type=str, choices=["sce1", "sce2"],
                    help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
parser.add_argument("--condition", type=str,
                    help="str, 'permute' or 'repeat'; whether or not to permute the second word list")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                    help="whether to run on cpu or cuda")
parser.add_argument("--input_filename", type=str,
                    help="str, the name of the .json file containing word lists")
parser.add_argument("--output_dir", type=str,
                    help="str, the name of folder to write the output_filename in")
parser.add_argument("--output_filename", type=str,
                    help="str, the name of the output file saving the dataframe")

argins = parser.parse_args()

# construct output file name and check that it existst
savedir = os.path.join(".", argins.output_dir)
assert os.path.isdir(savedir)                                 # check that the folder exists
outpath = os.path.join(savedir, argins.output_filename)

print("condition == {}".format(argins.condition))
print("scenario == {}".format(argins.scenario))

device = torch.device(argins.device if torch.cuda.is_available() else "cpu")  # declare device and paths

# initiate the sampler class from experiment.py module
s = Sampler(model=model, tokenizer=tokenizer, device=device)

# ===== DATASET ===== #
# load the word lists in .json files
with open(argins.input_filename) as f:
    print("Loading {} ...".format(argins.input_filename))
    stim = json.load(f)

# convert word lists to strings and permute the second one if needed
# add space at the string onset
word_list1 = [" " + ", ".join(l) + "." for l in stim]
if argins.condition == "permute":

    # This condition test for the effect of word order
    # Lists have the same words, but the word order is permuted
    # int the second one

    word_list2 = [" " + ", ".join(np.random.RandomState((543+j)*5).permutation(stim[j]).tolist()) + "."
                  for j in range(len(stim))]

elif argins.condition == "control":

    # This serves as a control conditions
    # Here list length is the only common factor between two lists

    print("Creating reverse control condition...")
    print("Assuming input list can be evenly split into 3 lists each of len(list)==20!")
    len3 = stim[0:20]
    len5 = stim[20:40]
    len10 = stim[40::]
    word_list2 = [" " + ", ".join(l) + "." for lst in (len3, len5, len10)
                              for l in reversed(lst)]

else:
    word_list2 = word_list1


# ===== COMPUTE PERPLEXITY ===== #

# call the wrapper function
# this one loops over prefixes and over prompts
output_list = s.run_perplexity(prefixes=prefixes[argins.scenario],
                               prompts=prompts[argins.scenario],
                               word_list1=word_list1,
                               word_list2=word_list2)

# ===== FORMAT AND SAVE OUTPUT ===== #

# convert the output to dataframe
dfout = []
counter = 1  # counter for trials

# loop over trials
for k, tup in enumerate(output_list):

    # convert the last two elements of the tuple to an array
    dftmp = pd.DataFrame(np.asarray([tup.token, tup.trialID, tup.positionID, tup.surp]).T,
                         columns=["token", "trialID", "positionID", "surp"])

    dftmp["ispunct"] = dftmp.token.isin([".", ":", ","])     # create punctuation info column
    dftmp['prefix'] = tup.prefix                             # add a column of prefix labels
    dftmp['prompt'] = tup.prompt                             # add a column of prompt labels
    dftmp["list_len"] = tup.list_len                         # add list length
    dftmp['stimID'] = counter
    dftmp['second_list'] = argins.condition                      # log condition of the second list

    dfout.append(dftmp)
    counter += 1

# put into a single df and save
dfout = pd.concat(dfout)

# save output
print("Saving {}".format(outpath))
dfout.to_csv(outpath, sep=",")
