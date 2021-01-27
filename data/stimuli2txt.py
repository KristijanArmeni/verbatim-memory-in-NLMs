

from stimuli import prompts, prefixes
import json
import argparse
from nltk import word_tokenize
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--json_filename", type=str)
parser.add_argument("--scenario", choices=["sce1"], type=str)
parser.add_argument("--permute", action="store_true")

args = parser.parse_args()

fname = args.json_filename

if args.permute:
    suffix = "permute"
else:
    suffix = "repeat"

outname = args.json_filename.replace(".json", "_{}.txt".format(suffix))
markersoutname = args.json_filename.replace(".json", "_{}_markers.txt".format(suffix))

trials = {
    "string": [],
    "markers": [],
    "prompt_len": []
}

with open(fname) as f:
    print("Loading {}".format(fname))
    stim = json.load(f)

input_lists = [", ".join(l) + "." for l in stim]

# select dict with correct prompts
prompts = prompts[args.scenario]

strings = []
for prefix_key in prefixes.keys():
    for prompt in prompts.keys():
        for j, l in enumerate(input_lists):

            l2 = None
            if args.permute:
                tmp = np.random.RandomState((543 + j) * 5).permutation(stim[j]).tolist()
                l2 = ", ".join(tmp) + "."
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
strings=map(lambda x:x+'\n', trials["string"])  # write newline characters
outfile.writelines(strings)
outfile.close()

# write the strings to output file name
print("Writing {}".format(markersoutname))
outfile = open(markersoutname, "w")
strings = map(lambda x, y:y+'\t'+x+'\n', trials["markers"], trials["prompt_len"])  # write newline characters
outfile.writelines(strings)
outfile.close()