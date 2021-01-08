"""
next_word_prediction.py defines the function and the script to run the next word prediction experiment

Use as:

next_word_prediction.py "model_class" "config"

"""
from experiment import Sampler
from stimuli import prefixes, prompts_word_prediction
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd

text_generation = True

# load the word lists
with open("./data/toronto.json") as f:
    stim = json.load(f)

# setup the models
models = {"gpt-2": (AutoModelForCausalLM, AutoTokenizer)}

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
s = Sampler(model=model, tokenizer=tokenizer)

# apply selection of conditions if needed
prefixes_orig = prefixes.copy()
prefixes = {key: prefixes_orig[key] for key in ["cpl"]}
conditions = ["cpl_any", "cpl_first"]
prompts_word_prediction_orig = prompts_word_prediction.copy()
prompts_word_prediction = {key: prompts_word_prediction_orig[key] for key in conditions}

# loop over the input lists

if text_generation:
    # with text generation, we always sample one token per position
    # this is dummy code
    num_tokens = [1]
else:
    num_tokens = [1, 5]

lst = []
for k in num_tokens:
    # loop over different prefixes:
    for prefix_key in prefixes.keys():
        for prompt in prompts_word_prediction.keys():
            for l in stim:
                input_string = prefixes[prefix_key] + " " + \
                               l + " " + \
                               prompts_word_prediction[prompt]
                print("Processing:\n{}".format(input_string))

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

                    # set parameters for model.generate()
                    num_beams = 4
                    top_p = 0.95
                    top_k = 60

                    text_len = 10  # the number of tokens to be generated

                    inputs = s.tokenizer.encode(input_string, add_special_tokens=False, return_tensors="pt")
                    max_len = len(inputs[0]) + text_len  # generate five tokens after the context is processed
                    outputs = s.model.generate(input_ids=inputs,
                                               max_length=max_len,
                                               do_sample=True,
                                               num_beams=num_beams,  # setting this to 1 avoids beam search
                                               top_p=top_p,
                                               top_k=top_k)

                    # fill in the output list
                    out_str = s.tokenizer.decode(outputs[0][-text_len::])

                    ishit = None
                    if (prompt == "cpl_any") or (prompt == "cnt"):
                        # check if any of the targets appear in the list
                        ishit = any([t in out_str.split() for t in targets])
                    elif prompt == "cpl_first":
                        # check if the first item appears anywhere in the output string
                        ishit = targets[0] in out_str.split()

                    lst.append({
                     "pref": prefix_key,
                     "prompt": prompt,
                     "len": len(targets),
                     "string": out_str,
                     "targets": targets,
                     "ishit": int(ishit),
                     "num_beams": num_beams,
                     "top_p": top_p,
                     "top_k": top_k}
                    )


dfout = pd.DataFrame(columns=list(lst[0].keys()))
out = dfout.append(lst, ignore_index=True)
if text_generation:
    outname = "./output/word_prediction_{}tok.txt".format(text_len)
else:
    outname = "./output/word_prediction_1tok.txt"
out.to_csv(path_or_buf=outname)