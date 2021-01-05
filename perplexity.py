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

# own modules
from experiment import model, tokenizer, Sampler
from stimuli import prefixes_perplexity, prompts_perplexity

s = Sampler(model=model, tokenizer=tokenizer)

# ===== DATASET ===== #
# load the word lists
with open("./data/toronto.json") as f:
    stim = json.load(f)

dataS["cond"] = "sem"
dataR["cond"] = "rnd"

# code a repetition condition
tmp = pd.DataFrame(columns=dataS.columns.tolist())
tmp["token"] = np.repeat("cheese", 10).tolist()
tmp['cond'] = 'rpt'

data = pd.concat([dataS, dataR, tmp])

# temp, redo the set index
num_sets = 5
data["setid"] = np.repeat(np.arange(1, num_sets+1), 10)
data.drop(["set"], axis=1, inplace=True)

# create some random word lists
wl = {"sem": [], "rnd": [], 'rpt': []}
for idx in data.setid.unique():

    # structure the list
    l = [s + ',' for s in data.loc[data.setid == idx, 'token'].tolist()]
    l[0] = l[0].capitalize()         # capitalize first letter
    l[-1] = l[-1].replace(',', '.')      # add full stop to the last item

    # write sentences to the stimuli
    wl[data.loc[data.setid == idx, 'cond'].iloc[0]].append(l)

# Construct dict with inputs
inputs = {key: {'sem': [], 'rnd': [], 'rpt': []} for key in prompts.keys()}
codes = {key: {'sem': [], 'rnd': [], 'rpt': []} for key in prompts.keys()}
for prompt in inputs.keys():
    for key in inputs[prompt].keys():
        for i in range(len(wl[key])):
            fullstring = " ".join([pref1, " ".join(wl[key][i]), prompts[prompt], " ".join(list(reversed(wl[key]))[i])])
            #string_code = [0 for j in pref1.split()] + [1 for e in wl[key][i]] + [2 for p in prompts[prompt].split()] + \
            #                                                                [3 for h in wl[key][i]]
            inputs[prompt][key].append(fullstring)
            #codes[prompt][key].append(string_code)

# ===== COMPUTE PERPLEXITY ===== #
out = {key: {"sem": [], "rnd": [], 'rpt': []} for key in prompts.keys()}

# loop over prompts
for prompt in out.keys():
    # loop over list conditions
    for list_condition in out[prompt].keys():
        # get perplexity
        for input_string in inputs[prompt][list_condition]:
            print("Computing ppl in condition {} prompt {}".format(prompt, list_condition))
            a, b, c = s.ppl(input_string=input_string, context_len=1024, stride=1, device="cpu")
            out[prompt][list_condition].append((a, b, c))

# convert output to dataframe
dfout = []
counter = 1
for prompt in out.keys():
    for key in out[prompt].keys():
        for k, tup in enumerate(out[prompt][key]):
            # convert the last two elements of the tuple to an array
            dftmp = pd.DataFrame(np.asarray(tup[1::]).T, columns=["ppl", "token"])
            dftmp['cond1'] = key
            dftmp['cond2'] = prompt
            dftmp['id'] = counter
            #dftmp['type'] = codes[prompt][key][k]  # list of 1, 2, 3, or 4 marking string type
            dfout.append(dftmp)
            counter += 1

# put into df and save
dfout = pd.concat(dfout)

dfout.to_csv("./perplexity3.txt", sep=",")
