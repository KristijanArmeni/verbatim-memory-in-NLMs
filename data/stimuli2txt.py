from stimuli import prompts, prefixes, prefixes_repeated_ngrams
import json
import argparse
from nltk import word_tokenize
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--json_filename", type=str)
parser.add_argument("--paradigm", type=str, choices=["with-context", "repeated-ngrams"])
parser.add_argument("--scenario_key", choices=["sce1", "sce1rnd", "sce2", "sce3"], 
                    default="sce1", type=str)
parser.add_argument("--condition", choices=["repeat", "permute", "control"],
                    default="repeat", type=str)
parser.add_argument("--list_only", action="store_true")

args = parser.parse_args()

fname = args.json_filename
input_dir = os.path.join(os.environ["homepath"], "project", "lm-mem", "src", "data")

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

# ==== LOAD .JSON FILES WITH WORD LISTS ==== #

if args.paradigm == "with-context":
    
    with open(fname) as f:
        print("Loading {}".format(fname))
        stim = json.load(f)
    
    # join word list items into strings for tokenization
    input_lists = [", ".join(l) + "." for l in stim]
    
elif args.paradigm == "repeated-ngrams":
    
    with open(fname) as f:
        print("Loading {}".format(fname))
        stim = json.load(f)
    
    # load the .json with distractor nouns
    distractor_file = os.path.join(input_dir, "ngram-distractors.json")
    with open(distractor_file) as f:
        print("Loading {}".format(distractor_file))
        dist = json.load(f)   

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

if args.paradigm == "with-context":

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


elif args.paradigm == "repeated-ngrams":
    
    
    def interleave(trg, dst):
        
        dst.append("") # add a dummy element at the end
        return [val for pair in zip(trg, dst) for val in pair]
    
    
    def code_tokens(tokens=None, markers=[1, 2], n_target_toks=None, n_dst_toks=None, repetitions=5):
        
        codes = list(np.tile(np.repeat(markers, [n_target_toks, n_dst_toks]), repetitions))
        
        # compute effective list length
        len_effect = len(tokens)
        len_full = (n_target_toks*repetitions) + ((n_dst_toks)*(repetitions))
        
        # there should always be just one distractor ngram too much, 
        # make sure it checks
        ngram_diff = len_full - len_effect
        assert ngram_diff == n_dst_toks
        
        # this process generates 1 ngram code too much, drop it
        del codes[-n_dst_toks:]
        assert len(codes) == len(tokens)
        
        return codes
    
    
    prefixes = prefixes_repeated_ngrams[args.scenario_key]
    
    ngram_sizes = list(stim.keys())
    distractor_sizes = list(dist.keys())
    
    # massive loops
    for prefix_key in prefixes:
        
        for ngram in ngram_sizes:
            
            for dst_size in distractor_sizes:
                
                targets = stim[ngram]
                distractors = dist[dst_size]
                
                # loop over ngram chunks for each a trial
                for trg in targets:
                    
                    for dst in distractors:
                        
                        interleaved = interleave(trg=trg, dst=dst)   
                        trial = ", ".join([", ".join(e) for e in filter(None, interleaved)]) + "."
                        
                        # tokenize words and then join the list back to a string with spaces
                        # apparently neural-complexity-master/main.py is used with tokenized input.
                        s1 = word_tokenize(prefixes[prefix_key])
                        s2 = word_tokenize(trial)
                        
                        # make exception for n-gram experiment
                        subparts = (s1, s2)
                        
                        # construct the string
                        string = " ".join([" ".join(part) for part in subparts])
                        
                        # create coding for parts of trials
                        # assume every token in ngram has punctuation
                        len_ngram_with_punct = int(ngram.split("-")[0])*2
                        len_dst_with_punct = int(dst_size.split("-")[-1])*2
                        
                        codes = code_tokens(tokens=s2, markers=[1, 2],
                                            n_target_toks=len_ngram_with_punct,
                                            n_dst_toks=len_dst_with_punct,
                                            repetitions=len(trg))
                        
                        codes_prefix = list(np.zeros(len(s1), dtype=int))
                        
                        # concatenate lists
                        labels_flattened = codes_prefix + codes
            
                        # quick check that all matches when spliting
                        assert len(string.split()) == len(labels_flattened)
            
                        trials["string"].append(string)
                        trials["markers"].append(str(labels_flattened))
                        trials['prompt_len'].append(ngram.split("-")[0])


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