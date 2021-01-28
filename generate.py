"""
generate.py is used to run the open ended text generation experiment

Example usage:

%run ./generate.py --do_sample "1" \
                               --num_beams "4" \
                               --temp "1" \
                               --top_p "0.95" \
                               --top_k "50" \
                               --max_tokens "15" \
                               --min_tokens "5" \
                               --input_list "nonsemantic" \
                               --savename "next_word_sample.csv"

"""
from experiment import Sampler
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'data'))  # this is needed for stimuli module to be found
from stimuli import prefixes, prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
import torch
import mlflow
import argparse
import datetime

text_generation = True

# collect input arguments
parser = argparse.ArgumentParser(description="generate.py")

parser.add_argument("--scenario", type=str, choices=["sce1"])
parser.add_argument("--do_sample", action="store_true",
                    help="bolean, passed as the value of <do_sample>"
                         "to model.generate() function")
parser.add_argument("--num_beams", type=int, default=1,
                    help="int, the value is passed to <num_beams> input arg in the call to model.generate()  ")
parser.add_argument("--temp", type=float, default=1.0,
                    help="float, temperature parameter the value is passed as <temperature>"
                         "input arg to model.generate()")
parser.add_argument("--top_p", type=float, default=9.0, help="")
parser.add_argument("--top_k", type=int, default=1000, help="")
parser.add_argument("--max_tokens", type=int, help="")
parser.add_argument("--min_tokens", type=int, help="")
parser.add_argument("--input_list", type=str,
                    help="string, used as a control flag to load in the corresponding .json file")
parser.add_argument("--output_dir", type=str,
                    help="folder to whic output file is written")
parser.add_argument("--savename", type=str,
                    help="string, specifies the name of the output file")
parser.add_argument("--device", type=str, default="cpu", choices=["gpu", "cpu"],
                    help="whether to use GPU or CPU")

argins = parser.parse_args()

# turn of irrelevant params if beam search is performed
if not argins.do_sample:
    argins.top_k = None
    argins.top_p = None
    argins.temp = None

print("do_sample={} setting the following params as:".format(argins.do_sample))
print("top_p={}".format(argins.top_p))
print("top_k={}".format(argins.top_k))
print("num_beams={}".format(argins.num_beams))

params = {
    "do_sample": argins.do_sample,  # convert string to int, and int to bool
    "num_beams": argins.num_beams,
    "temperature": argins.temp,
    "top_p": argins.top_p,
    "top_k": argins.top_k,
    "max_tokens": argins.max_tokens,   # the upper bound on the number of tokens to be generated
    "min_tokens": argins.min_tokens  # the lower bound on the number of tokens to be generated
}

# load the word lists
fname = None
if argins.input_list == "random":
    fname = "./data/random_lists.json"
elif argins.input_list == "categorized":
    fname = "./data/categorized_lists.json"
with open(fname) as f:
    print("Loading {}".format(fname))
    stim = json.load(f)

# convert list to strings
input_lists = [", ".join(l) + "." for l in stim]

# setup the models
models = {"gpt-2": (AutoModelForCausalLM, AutoTokenizer)}

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
s = Sampler(model=model, tokenizer=tokenizer, device=argins.device)

device = torch.device(argins.device if torch.cuda.is_available() else "cpu")  # declare device and paths

model.to(argins.device)

if text_generation:
    # with text generation, we always sample one token per position
    # this is dummy code
    num_tokens = [1]
else:
    num_tokens = [1, 5]

prefixes = prefixes[argins.scenario]
prompts = prompts[argins.scenario]

lst = []
count = 1
total = len(input_lists)*len(prefixes.keys())*len(prompts.keys())

# set mlflow business
mlflow.set_experiment(experiment_name="open_ended_text_generation")

with mlflow.start_run():

    mlflow.set_tag("date-time", datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    mlflow.log_params(params)

    for k in num_tokens:
        # loop over different prefixes:
        for prefix_key in prefixes.keys():
            for prompt in prompts.keys():
                for l in input_lists:

                    print("Trial {}/{}".format(count, total))

                    input_string = prefixes[prefix_key] + " " + \
                                   l + " " + \
                                   prompts[prompt]

                    # convert string back to list, remove comas and periods
                    targets = l.replace(",", "").replace(".", "").split()

                    # execute this block for word prediction task
                    if not text_generation:

                        logits = s.get_logits(input_string=input_string)
                        # collect the next token prediction and its probability
                        next_token_probs, next_token_ids = s.get_probs(logits=logits, k=50, p=0.9, n_tokens=k)

                        next_tokens = s.tokenizer.decode(next_token_ids[0])

                        print("Prediction at k == {}: {}".format(k, next_tokens))

                        # score based on criteria
                        ishit = None
                        if ("starts with" in prompt) or ("the first item" in prompt) and k == 1:
                            if k == 1:
                                ishit = int(next_tokens == targets[0])
                            elif k > 1:
                                ishit = any(item in next_tokens.split() for item in targets)
                        else:  # otherwise count hit if the token is anywhere
                            ishit = any(item in next_tokens.split() for item in targets)

                        probs = None
                        if k == 1:
                            probs = str(next_token_probs[0][0].tolist())
                        elif k > 1:
                            probs = str(next_token_probs[0].tolist())

                        lst.append(
                            {
                                "pref": prefix_key,
                                "prompt": prompt,
                                "len": len(l),
                                "prediction": next_tokens,
                                "prob": probs,
                                "ishit": ishit,
                                "top-k": k
                            }
                        )

                    # execute this block for text generation task
                    if text_generation:

                        inputs = s.tokenizer.encode(input_string,
                                                    add_special_tokens=False,
                                                    return_tensors="pt").to(device)

                        max_len = len(inputs[0]) + argins.max_tokens

                        # set rng seed for reproducibility
                        torch.manual_seed(54321)

                        if argins.do_sample:
                            outputs = s.model.generate(input_ids=inputs,
                                                       max_length=max_len,
                                                       min_length=len(inputs[0]) + argins.min_tokens,     # setting this to 1 avoids beam search
                                                       bad_words_ids=[[366], [1085]],                     # block generation of '"' and ':'
                                                       do_sample=argins.do_sample,
                                                       temperature=argins.temp,
                                                       top_k=argins.top_k,
                                                       top_p=argins.top_p
                                                       )
                        else:
                            outputs = s.model.generate(input_ids=inputs,
                                                       max_length=max_len,
                                                       min_length=len(inputs[0]) + argins.min_tokens,
                                                       bad_words_ids=[[366], [1085]],                     # block generation of '"' and ':'
                                                       num_beams=argins.num_beams,
                                                       )

                        # fill in the output list, go back max_tokens
                        out_str = s.tokenizer.decode(outputs[0][-argins.max_tokens::])
                        prompt_str = s.tokenizer.decode(outputs[0][0:len(inputs[0])])
                        # grab the string generated after ":", in case it was shorter than max_tokens
                        gen_str = out_str.split(":")[-1]

                        print("generated string:\n {} {}".format(prompt_str, gen_str))
                        print("targets: {}".format(targets))

                        # check if any of the targets appear in the list
                        output_list = [s.strip(".|,") for s in out_str.split()]
                        hits = [t in output_list for t in targets]
                        prop = sum(hits)/len(hits)

                        print("{} hits out of {}".format(sum(hits), len(hits)))

                        lst.append({
                         "pref": prefix_key,
                         "prompt": prompt,
                         "len": len(targets),
                         "string": output_list,
                         "targets": targets,
                         "list": argins.input_list,
                         "n_correct": sum(hits),
                         "p_correct": prop,
                         "do_sample": argins.do_sample,
                         "num_beams": argins.num_beams,
                         "top_p": argins.top_p,
                         "top_k": argins.top_k,
                         "temp": argins.temp}
                        )

                    count += 1

dfout = pd.DataFrame(columns=list(lst[0].keys()))
out = dfout.append(lst, ignore_index=True)

outname = os.path.join(argins.output_dir, argins.savename)


#with open("outputs/word_prediction_{}tok_log.txt".format(argins.max_tokens), "w") as f:
#    f.write("hello world!")
#mlflow.log_artifacts(local_path="outputs/outputs/word_prediction_{}tok_log.txt")

out.to_csv(path_or_buf=outname)

