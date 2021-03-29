
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F
from tqdm import trange
from typing import Dict, List
import numpy as np
from collections import namedtuple

# setup the model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')


class Exp(object):
    """
    Exp() class contains wrapper methods to run experiments with transformer models.
    """
    def __init__(self, model, tokenizer, device, config):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_len = config["context_len"]

        self.model.to(device)

    def get_logits(self, input_string):
        
        inp_ids = self.tokenizer.encode(input_string, return_tensors="pt")

        # take the output logits of the model
        with torch.no_grad():
            logits = self.model(inp_ids)[0]  # tuple of shape = (logits, layer and attention output)

        return logits
            
    def get_probs(self, logits, n_tokens=5, k=10^3, p=0.9):
        
        """get_probs(self, logtis, k, p) is a convenience fun wrapping around 
        top_k_top_p_filtering(logits, top_k, top_p), calling F.softmax() on the output and sampling from   there
        """
        
        # take only the top 10 logits
        filtered_logits = top_k_top_p_filtering(logits[:, -1, :], top_k=k, top_p=p)

        # apply the softmax to the truncated distribution
        probs = F.softmax(filtered_logits, dim=-1)

        # now sample n tokens
        # next_tokens = torch.multinomial(probs, num_samples=n_tokens, replacement=False)

        token_prob, token_ids = torch.topk(x=probs, dim=1, k=n_tokens, sorted=True)

        return token_prob, token_ids

    def uniform_att(self, model):

        # access transformer blocks
        for block in self.model.transformer.h:

            Q, K, V = block.attn.c_attn.weight

        return model

    # loop over input list
    def ppl(self, input_ids, context_len, stride, device):
        """
        method for computing token-by-token negative log likelihood on input_ids
        taken from: https://huggingface.co/transformers/perplexity.html
        """

        llh = []  # variable storing token-by-token neg ll
        id = []   # variable storing token strings to have along with -ll in the output

        for i in trange(0, input_ids.size(1), stride, desc="Computing perplexity: "):

            # define the start and endpoints of input indices for current loop
            begin_loc = max(i + stride - context_len, 0)  # define the non-negative onset location
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # select the current input index span
            sel_input_ids = input_ids[:, begin_loc: end_loc].to(device)

            # define target labels, use input ids as target outputs
            target_ids = sel_input_ids.clone()
            target_ids[:, :-trg_len] = -100  # ignore the tokens that were used for context when computing the loss

            # set model to evaluation mode
            self.model.eval()

            # get model output
            with torch.no_grad():

               # compute log likelihood
               outputs = self.model(sel_input_ids, labels=target_ids)  # this returns avg log likelihood over sequence
               log_likelihood = outputs[0] * trg_len  # not sure about this multiplication here (undoing averaging?)

               llh.append(log_likelihood.tolist())
               toks = self.tokenizer.decode(target_ids[0][-stride::])
               id.append(toks)  # store the last token (target_id)

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppl = torch.exp(torch.tensor(np.nansum(llh)) / end_loc)
        print("Done")
        return ppl, llh, id

    def run_perplexity(self, prefixes: Dict, prompts: Dict, word_list1: List, word_list2: List) -> List:
        """
        experiment.run() will loop over prefixes, prompts, and word_lists and run the Sampler on them
        It returns a list of tuples.
        """

        lst = []
        count = 1

        # store output in named tuple for more controle
        Output = namedtuple(typename="Output",
                            field_names=["ppl", "surp", "token", "trialID", "positionID", "list_len",
                                         "prefix", "prompt"])

        # quick check that the lengths match
        assert len(word_list1) == len(word_list2)

        total = len(prefixes.keys())*len(prompts.keys())*len(word_list1)

        # loop over different prefixes:
        for prefix_key in prefixes.keys():

            # loop over prompts
            for prompt_key in prompts.keys():

                # loop over trials
                for i in range(len(word_list1)):

                    # tokenize strings separately to be able to construct markers for prefix, word lists etc.
                    i1 = self.tokenizer.encode("<|endoftext|> " + prefixes[prefix_key], return_tensors="pt")   # prefix IDs, add eos token
                    i2 = self.tokenizer.encode(word_list1[i], return_tensors="pt")                             # word list IDs
                    i3 = self.tokenizer.encode(prompts[prompt_key], return_tensors="pt")                       # prompt IDs
                    i4 = self.tokenizer.encode(word_list2[i] + " <|endoftext|>", return_tensors="pt")

                    # compose the input ids tensors
                    input_ids = torch.cat((i1, i2, i3, i4), dim=1)

                    # construct IDs for prefix, word lists and individual tokens
                    # useful for data vizualization etc.
                    trials = []
                    positions = []
                    for j, ids in enumerate((i1, i2, i3, i4)):
                        tmp = np.zeros(shape=ids.shape[1], dtype=int)  # code the trial structure
                        tmp[:] = j
                        tmp2 = np.arange(ids.shape[1])                 # create token position index
                        trials.append(tmp)
                        positions.append(tmp2)

                    print("counter: {}/{}".format(count, total))

                    # this returns surprisal (neg log ll)
                    a, b, c = self.ppl(input_ids=input_ids.to(self.device), context_len=self.context_len, stride=1,
                                       device=self.device)

                    # store the output tuple and
                    lst.append(Output(ppl=a, surp=b, token=c,                            # perplexity output
                                      trialID=np.concatenate(trials).tolist(),           # trial structure
                                      positionID=np.concatenate(positions).tolist(),     # position index
                                      list_len=len([e.strip().replace(".", "") for e in word_list1[i].split(",")]),
                                      prefix=prefix_key,
                                      prompt=prompt_key))

                    count += 1  # increase counter for feedback

        return lst
