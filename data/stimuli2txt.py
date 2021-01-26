

from stimuli import prompts_perplexity, prefixes
import json
import argparse
from nltk import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--json_filename", type=str)

args = parser.parse_args()

fname = args.json_filename
outname = args.json_filename.replace(".json", ".txt")
markersoutname = args.json_filename.replace(".json", "_markers.txt")

trials = {
    "string": [],
    "markers": [],
}

with open(fname) as f:
    print("Loading {}".format(fname))
    stim = json.load(f)

input_lists = [", ".join(l) + "." for l in stim]

strings = []
for prefix_key in prefixes.keys():
    for prompt in prompts_perplexity.keys():
        for l in input_lists:

            # tokenize each string separately, then make strings again
            # and concatenate the strings together
            s1 = word_tokenize(prefixes[prefix_key])
            s2 = word_tokenize(l)
            s3 = word_tokenize(prompts_perplexity[prompt])
            s4 = word_tokenize(l)

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

for fname, values in zip((outname, markersoutname), (trials["string"], trials["markers"])):
    # write the strings to output file name
    print("Writing {}".format(fname))
    outfile = open(fname, "w")
    strings=map(lambda x:x+'\n', values)  # write newline characters
    outfile.writelines(strings)
    outfile.close()