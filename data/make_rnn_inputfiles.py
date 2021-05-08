from stimuli import prompts, prefixes
import json
import argparse
from nltk import word_tokenize
import numpy as np
import os, sys

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

if  "linux" in sys.platform:
    input_dir = os.path.join(os.environ["HOME"], "code", "lm-mem", "data")
elif  "win" in sys.platform:
    input_dir = os.path.join(os.environ["homepath"], "project", "lm-mem", "src", "data")

if args.list_only: 
    scenario = "nosce" 
else: 
    scenario = args.scenario_key


# ===================== #
# ===== FUNCTIONS ===== #
# ===================== #

def sample_indices_by_group(groups, seed):
    
    out_ids = np.zeros(groups.shape, dtype=int)
    indices = np.arange(groups.size)
    rng = np.random.RandomState(seed)
    ignore_id = -1
    
    # number of selected samples must mach size of one group
    sample_size = np.sum(groups == 0)
    
    for group in np.unique(groups):
        
        # choose indices not from current group and not the ones already sampled
        candidate_pool = indices[(groups != group) & (indices != ignore_id)]
        
        sel_ids = rng.choice(a=candidate_pool, size = sample_size)
        out_ids[groups == group] = sel_ids
        
        # mark already selected indices
        indices[sel_ids] = ignore_id
        
    return out_ids


def tokenize_and_concat(prefix, list1, context, list2):
    
    """
    tokenize_and_concat() is just a wrapper function to make
    the loop below slightly more readable.
    
    """
    
    # tokenize words and then join the list back to a string with spaces
    # apparently neural-complexity-master/main.py is used with tokenized input.
    s1 = word_tokenize(prefix)
    s2 = word_tokenize(list1)
    s3 = word_tokenize(context)
    s4 = word_tokenize(list2)

    # make exception for n-gram experiment
    subparts = (s1, s2, s3, s4)
    
    # create coding for parts of trials
    codes = [[i]*len(tokens) for i, tokens in enumerate(subparts)]
    codes_flattened = [item for sublist in codes for item in sublist]
    
    # construct the string
    string = " ".join([" ".join(part) for part in subparts])

    # quick check that all matches when spliting
    assert len(string.split()) == len(codes_flattened)
    
    return string, codes_flattened


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
    if n_dst_toks != 0:
        del codes[-n_dst_toks:]
    assert len(codes) == len(tokens)
    
    return codes


def ensure_list2_notequal(list1, list2, start_seed, seed_increment):
    
    """
    new_list2 = ensure_list2_notequal(list1, list2, start_seed, seed_increment)
    
    Helper function that ensures that all elements in list2 are not
    equal to elements in list1 by iteratively applying new perumtations
    with a new start_seed, incremented by seed_increment.
    """
    
    are_equal = [[t1 == t2] for t1, t2 in zip(list1, list2)]
    
    seed = start_seed
    
    # if lists are already disjoint, just return list2
    if ~np.any(are_equal):
        list2_new = list2
    
    else:
        # do this until elements of list1 and list2 are not equal
        while np.any(are_equal):
            
            rng = np.random.RandomState(seed)
            
            # create new permutations for l2 that are still equal
            list2_new = [rng.permutation(l2).tolist() if l1 == l2 else l2 
                         for l1, l2 in zip(list1, list2)]
            
            # update the criterion condition (none should be equal)
            are_equal = [[l1 == l2] for l1, l2 in zip(list1, list2_new)]
            
            # update seed for a new try
            seed += seed_increment
    
    return list2_new

# ===================== #
# ===== LOAD DATA ===== #
# ===================== #

if args.paradigm == "with-context":
    
    with open(fname) as f:
        print("Loading {}".format(fname))
        stim = json.load(f)
    
elif args.paradigm == "repeated-ngrams":
    
    fname = os.path.join(input_dir, "ngram-random.json")
    with open(fname) as f:
        print("Loading {}".format(fname))
        stim = json.load(f)
    
    # load the .json with distractor nouns
    distractor_file = os.path.join(input_dir, "ngram-distractors.json")
    with open(distractor_file) as f:
        print("Loading {}".format(distractor_file))
        dist = json.load(f)   

word_lists2 = None
if args.condition == "control":

    print("Creating control condition...")
    
    n_items_per_group = 10
    n_groups = len(stim["n10"])//n_items_per_group
    groups = np.repeat(np.arange(0, n_groups), n_items_per_group)
    
    ids = sample_indices_by_group(groups=groups, seed=12345)
    
    word_lists2 = {key: np.asarray(stim[key])[ids].tolist() for key in stim.keys()}
    
    # make sure control tokens do not appear in the target lists
    for k in stim.keys():
        assert ~np.any([set(t1).issubset(set(t2)) 
                        for t1, t2 in zip(stim[k], word_lists2[k])])
        
elif args.condition == "permute":
    
    print("Creating permute condition")
    
    # This condition test for the effect of word order
    # Lists have the same words, but the word order is permuted
    # int the second one
    word_lists2 = {key: [np.random.RandomState((543+j)*5).permutation(stim[key][j]).tolist()
                  for j in range(len(stim[key]))]
                  for key in stim.keys()}
    
    # some lists in word_lists2 may end up the equal to word_lists1 by chance,
    # check for that and reshuffle if needed
    word_lists1=stim
    for list_size in word_lists2.keys():
        
        word_lists2[list_size] = ensure_list2_notequal(list1=word_lists1[list_size],
                                                       list2=word_lists2[list_size],
                                                       start_seed=123,
                                                       seed_increment=10)

    # make sure permuted lists are not equal to target lists
    for k in stim.keys():
        assert ~np.any([[t1 == t2] for t1, t2 in zip(word_lists1[k], word_lists2[k])])

elif args.condition == "repeat":
    word_lists2 = stim
 
# select dict with correct prompts and prefixes
prompts = prompts[args.scenario_key]
prefixes = prefixes[args.scenario_key]

all_prefixes = prefixes.keys()
all_prompts = prompts.keys()

# if ngram experiment only use one scenario level
if "ngram" in args.json_filename:
    all_prefixes = list(prefixes.keys())[0:1]
    all_prompts = list(prompts.keys())[0:1]


# ========================= #
# ===== MAIN ROUTINES ===== #
# ========================= #

outname = args.json_filename.replace(".json", "_{}_{}.txt".format(scenario, args.condition))
markersoutname = args.json_filename.replace(".json", "_{}_{}_markers.txt".format(scenario, args.condition))

# use different marker labels for two experiments

if args.paradigm == "repeated-ngrams":
    marker_keys = ["string", "markers", "stimid",  "list_len", "dist_len"]
elif args.paradigm == "with-context":
    marker_keys = ["string", "markers", "stimid", "list_len", "prompt_len"]

trials = {key: [] for key in marker_keys}


# contextualized paradigm
if args.paradigm == "with-context":

    strings = []
    for prefix_key in all_prefixes:
        for prompt_key in all_prompts:
            for list_size in stim.keys():
                
                current_list = stim[list_size]
                target_lists = word_lists2[list_size]
                
                for j, l in enumerate(current_list):
                    
                    l1 = ", ".join(current_list[j]) + "."
                    l2 = ", ".join(target_lists[j]) + "."
        
                    string, markers = tokenize_and_concat(prefix=prefixes[prefix_key], 
                                                          list1=l1, 
                                                          context=prompts[prompt_key], 
                                                          list2=l2)
        
                    trials[marker_keys[0]].append(string)
                    trials[marker_keys[1]].append(str(markers))
                    trials[marker_keys[2]].append(str(j))
                    trials[marker_keys[3]].append(str(list_size.strip("n")))
                    trials[marker_keys[4]].append(prompt_key)


elif args.paradigm == "repeated-ngrams":
    
    
    ngram_sizes = list(stim.keys())
    distractor_sizes = ["n0"] + list(dist.keys()) # conde the non-interleaved condition as well
        
    for ngram in ngram_sizes:
        
        for dst_size in distractor_sizes:
            
            targets = stim[ngram]
            distractors = [None]
            if dst_size != "n0":
                distractors = dist[dst_size]
            
            
            # loop over ngram chunks for each a trial
            for i, trg in enumerate(targets):
                
                for dst in distractors:
                    
                    nouns = trg
                    
                    # if there are distractors, interleave them
                    if dst is not None:
                        nouns = interleave(trg=trg, dst=dst) 
                    
                    trial = ", ".join([", ".join(e) for e in filter(None, nouns)]) + "."
                    
                    # tokenize words and then join the list back to a string with spaces
                    # apparently neural-complexity-master/main.py is used with tokenized input.
                    #s1 = word_tokenize(prefixes[prefix_key])
                    toks = word_tokenize(trial)
                    
                    # make exception for n-gram experiment
                    #subparts = (s1, s2)
                    
                    # construct the string
                    fullstring = " ".join(toks)
                    
                    # create coding for parts of trials
                    # assume every token in ngram has punctuation
                    len_ngram_with_punct = int(ngram.strip("n"))*2
                    len_dst_with_punct = int(dst_size.strip("n"))*2
                    
                    codes = code_tokens(tokens=toks, markers=[1, 2],
                                        n_target_toks=len_ngram_with_punct,
                                        n_dst_toks=len_dst_with_punct,
                                        repetitions=len(trg))
                    
                    #codes_prefix = list(np.zeros(len(s1), dtype=int))
                    
                    # concatenate lists
                    #labels_flattened = codes_prefix + codes
        
                    # quick check that all matches when spliting
                    assert len(toks) == len(codes)
        
                    trials[marker_keys[0]].append(fullstring)
                    trials[marker_keys[1]].append(str(codes))
                    trials[marker_keys[2]].append(str(i))
                    trials[marker_keys[3]].append(ngram.strip("n"))
                    trials[marker_keys[4]].append(dst_size.strip("n"))


outdir = os.path.join(input_dir, "rnn_input_files")

# write the strings to output file name
print("Writing {}".format(os.path.join(outdir, outname)))
outfile = open(os.path.join(outdir, outname), "w")
strings = map(lambda x:x+'\n', trials["string"])  # write newline characters
outfile.writelines(strings)
outfile.close()

# write the strings to output file name
print("Writing {}".format(os.path.join(outdir, markersoutname)))
outfile = open(os.path.join(outdir, markersoutname), "w")

header = "\t".join([marker for marker in marker_keys[1::]]) + "\n"
outfile.write(header)

joinstrings = lambda a, b, c, d: a + '\t' + b + '\t' + c + '\t' + d + '\n'
rows = map(joinstrings,
           trials[marker_keys[1]], trials[marker_keys[2]], trials[marker_keys[3]], trials[marker_keys[4]])  # write newline characters
outfile.writelines(rows)
outfile.close()
