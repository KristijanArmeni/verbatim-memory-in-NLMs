

from stimuli import prompts, prefixes
import json
import argparse
from nltk import word_tokenize
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--json_filename", type=str)
parser.add_argument("--scenario", choices=["sce1", "sce1rnd", "sce2", "rpt"], type=str)
parser.add_argument("--condition", choices=["repeat", "permute", "control"], type=str)

args = parser.parse_args()

fname = args.json_filename

outname = args.json_filename.replace(".json", "_{}_{}.txt".format(args.scenario, args.condition))
markersoutname = args.json_filename.replace(".json", "_{}_{}_markers.txt".format(args.scenario, args.condition))

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
    print("Assuming input list can be evenly split into 3 lists each of len(list)==20!")
    len3 = stim[0:20]
    len5 = stim[20:40]
    len10 = stim[40::]
    stim_reversed = [l for lst in (len3, len5, len10)
                       for l in reversed(lst)]

input_lists = [", ".join(l) + "." for l in stim]

# select dict with correct prompts and prefixes
prompts = prompts[args.scenario]
prefixes = prefixes[args.scenario]

strings = []
for prefix_key in prefixes.keys():
    for prompt in prompts.keys():
        for j, l in enumerate(input_lists):

            l2 = None
            if args.condition == "permute":
                tmp = np.random.RandomState((543 + j) * 5).permutation(stim[j]).tolist()
                l2 = ", ".join(tmp) + "."
            elif args.condition == "control":
                l2 = ", ".join(stim_reversed[j]) + "."
            else:
                l2 = l

            # tokenize each string separately, then make strings again
            # and concatenate the strings together
            s1 = word_tokenize(prefixes[prefix_key])
            s2 = word_tokenize(l)
            s3 = word_tokenize(prompts[prompt])
            s4 = word_tokenize(l2)

            # create coding for parts of trials
            labels = [[i]*len(tokens) for i, tokens in enumerate((s1, s2, s3, s4))]
            labels_flattened = [item for sublist in labels for item in sublist]
            # tokenize words and then join the list back to a string with spaces
            # apparently neural-complexity-master/main.py is used with tokenized input.

            string = " ".join(s1) + " " + \
                     " ".join(s2) + " " + \
                     " ".join(s3) + " " + \
                     " ".join(s4)

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