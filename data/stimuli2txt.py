from stimuli import prompts, prefixes
import json
import argparse
from nltk import word_tokenize
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--json_filename", type=str)
parser.add_argument("--scenario_key", choices=["sce1", "sce1rnd", "sce2", "rpt"], 
                    default="sce1", type=str)
parser.add_argument("--condition", choices=["repeat", "permute", "control"],
                    default="repeat", type=str)
parser.add_argument("--list_only", action="store_true")

args = parser.parse_args()

fname = args.json_filename

if args.list_only: 
    scenario = "nosce" 
else: 
    scenario = args.scenario_key

outname = args.json_filename.replace(".json", "_{}_{}.txt".format(scenario, args.condition))
markersoutname = args.json_filename.replace(".json", "_{}_{}_markers.txt".format(scenario, args.condition))

trials = {
    "string": [],
    "markers": [],
    "prompt_len": []
}

with open(fname) as f:
    print("Loading {}".format(fname))
    stim = json.load(f)

stim_reversed = None
if args.condition == "control":

    print("Creating reverse control condition...")
    print("Assuming there are 3 levels of input lists can be evenly split into 3 lists each of len(list)==20!")
    
    # test that we can divide by 3, 
    # otherwise we sth has changed in the input lists
    assert len(stim) % 3 == 0
    
    list_len = len(stim)//3
    
    len3 = stim[0:list_len]
    len5 = stim[list_len:list_len*2]
    len10 = stim[list_len*2::]
    
    stim_reversed = [l for lst in (len3, len5, len10)
                       for l in reversed(lst)]

input_lists = [", ".join(l) + "." for l in stim]

# select dict with correct prompts and prefixes
prompts = prompts[args.scenario_key]
prefixes = prefixes[args.scenario_key]

all_prefixes = prefixes.keys()
all_prompts = prompts.keys()

# if ngram experiment only use one scenario level
if "ngram" in args.json_filename:
    all_prefixes = list(prefixes.keys())[0:1]
    all_prompts = list(prompts.keys())[0:1]

# ===== CREATE RNN INPUT STRINGS ===== #
strings = []
for prefix_key in all_prefixes:
    for prompt in all_prompts:
        for j, l in enumerate(input_lists):

            l2 = None
            if args.condition == "permute":
                tmp = np.random.RandomState((543 + j) * 5).permutation(stim[j]).tolist()
                l2 = ", ".join(tmp) + "."
            elif args.condition == "control":
                l2 = ", ".join(stim_reversed[j]) + "."
            else:
                l2 = l

            # tokenize words and then join the list back to a string with spaces
            # apparently neural-complexity-master/main.py is used with tokenized input.
            s1 = word_tokenize(prefixes[prefix_key])
            s2 = word_tokenize(l)
            s3 = word_tokenize(prompts[prompt])
            s4 = word_tokenize(l2)

            # make exception for n-gram experiment
            subparts = (s1, s2, s3, s4)
            
            # create coding for parts of trials
            labels = [[i]*len(tokens) for i, tokens in enumerate(subparts)]
            labels_flattened = [item for sublist in labels for item in sublist]
            
            # construct the string
            string = " ".join([" ".join(part) for part in subparts])

            # quick check that all matches when spliting
            assert len(string.split()) == len(labels_flattened)

            trials["string"].append(string)
            trials["markers"].append(str(labels_flattened))
            trials['prompt_len'].append(prompt)

# write the strings to output file name
print("Writing {}".format(outname))
outfile = open(outname, "w")
strings = map(lambda x:x+'\n', trials["string"])  # write newline characters
outfile.writelines(strings)
outfile.close()

# write the strings to output file name
print("Writing {}".format(markersoutname))
outfile = open(markersoutname, "w")
strings = map(lambda x, y:y+'\t'+x+'\n', trials["markers"], trials["prompt_len"])  # write newline characters
outfile.writelines(strings)
outfile.close()