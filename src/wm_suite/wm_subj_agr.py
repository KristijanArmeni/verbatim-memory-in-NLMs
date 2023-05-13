
import os
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Dict, Tuple
import torch
import logging
from tqdm import tqdm
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(message)s")

LINZEN2016 = "/scratch/ka2773/project/lm-mem/sv_agr/linzen2016/linzen2016_english.json"
LAKRETZ2021_SHORT = "/scratch/ka2773/project/lm-mem/sv_agr/lakretz2021/short_nested_inner_english.json"
LAKRETZ2021_LONG = "/scratch/ka2773/project/lm-mem/sv_agr/lakretz2021/long_nested_inner_english.json"


WIEGHTS_DIR = "/scratch/ka2773/project/lm-mem/output/ablation/zero_qk"
ABLATION_KEYS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "01", "23", "0123", "89", "1011", "891011", "all"]

MODEL_WEIGTHS = {f"ablate-{infix}": os.path.join(WIEGHTS_DIR, f"gpt2_ablate-{infix}-all.pt") for infix in ABLATION_KEYS}
MODEL_WEIGTHS["unablated"] = "gpt2"
MODEL_WEIGTHS["rand-init"] = "rand-init"


def read_json_data(path: str) -> Dict:  

    logging.info(f"Loading {path}...")
    with open(path, 'r') as fh:

        d = json.load(fh)

    return d['examples']


def get_inputs_and_targets(examples_list: Dict, comment: str) -> Tuple:

    has_comment = False
    if "comment" in examples_list[0].keys():
        has_comment = True

    # lakretz dataset comes in 4 versions, make it possible to only use a subsest of versions by adding <comment>
    if has_comment & (comment != ""):
        inputs = [e['input'].strip() for e in examples_list if e['comment'] == comment]
        targets = [e['target_scores'] for e in examples_list if (e['comment'] == comment) & (len(e['target_scores']) == 2)]
    else:
        # linzen dataset has whole sentences as targets, need to do .split() first
        inputs = [e['input'].strip() for e in examples_list]
        targets = [{key.split(' ')[0]: e['target_scores'][key] for key in e['target_scores'].keys()} for e in examples_list if (len(e['target_scores']) == 2)]
        
    logging.info(f"Number of examples == {len(inputs)}")

    return inputs, targets


def get_agreement_score(logits: torch.Tensor, targets: Tuple, tokenizer) -> int:

    probs = torch.softmax(logits, 0)  # convert logits to probabilities 

    tar = list(targets.keys())

    # make sure we add space as a word marker to the string itself
    # otherwise, the BPE tokenizer assumes the string is attached to another strin and BPE splits it
    corr_id = tokenizer.encode(" " + tar[0])
    incorr_id = tokenizer.encode(" " + tar[1])

    # if there's more than to IDs, it means the verb was BPE-split, skip it
    # there are not that many that need to be skipped
    if (len(corr_id) > 1) or (len(incorr_id) > 1):
        #logging.info("Skipping due to BPE split")
        out = {}

    else:
        strings = tuple(tokenizer.decode(id) for id in [corr_id, incorr_id])

        prob_correct = probs[corr_id].item()
        prob_incorrect = probs[incorr_id].item()
        
        #print(strings)
        #print((prob_correct, prob_incorrect))
        out = {key: v for key, v in zip(strings, (prob_correct, prob_incorrect))}
        out['res'] = int(prob_correct > prob_incorrect)
        out['max'] = torch.max(probs).cpu().item()
        out['corr_percentile'] = (torch.sum(probs > prob_correct).cpu().item())/len(probs)

    return out


def collect_agreement_scores(model, inputs, targets, tokenizer):

    scores = []

    for inp, tgt in tqdm(zip(inputs, targets), total=len(inputs)):

        model.eval()
        o = model(input_ids=inp)

        s = get_agreement_score(o.logits[-1, :], targets=tgt, tokenizer=tokenizer)

        scores.append(s)

    return scores


def get_accuracy(scores):

    res = [e['res'] for e in scores if bool(e)]
    return round(sum(res)/len(res), 4)


def run_experiment(dataset_path, model, tokenizer, comment, experiment_id):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    examples = read_json_data(dataset_path)

    if comment:
        inputs, targets = get_inputs_and_targets(examples_list=examples, comment=comment)
    else:
        inputs, targets = get_inputs_and_targets(examples_list=examples, comment=None)

    def inputs2tensor(inputs):
        return [torch.LongTensor(tokenizer.encode(x)).to(device) for x in inputs]

    scores = collect_agreement_scores(model.to(device), 
                                      inputs=inputs2tensor(inputs), 
                                      targets=targets,
                                      tokenizer=tokenizer)

    output = {}
    output["scores"] = scores
    output["accuracy"] = get_accuracy(scores)
    output["dataset"] = dataset_path
    output["type"] = comment
    output['id'] = experiment_id

    return output


def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir")

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    # set seed for reproducibility
    torch.manual_seed(54321)

    dataset_dict = {
        "linzen2016": {'path': LINZEN2016, 'comment': None}, 
        "lakretz2021_S_A": {'path': LAKRETZ2021_SHORT, 'comment': "singular_singular"},
        "lakretz2021_S_B": {'path': LAKRETZ2021_SHORT, 'comment': "singular_plural"},
        "lakretz2021_S_C": {'path': LAKRETZ2021_SHORT, 'comment': "plural_singular"},
        "lakretz2021_S_D": {'path': LAKRETZ2021_SHORT, 'comment': "plural_plural"},
        "lakretz2021_L_A": {'path': LAKRETZ2021_LONG, 'comment': "singular_singular_singular"},
        "lakretz2021_L_B": {'path': LAKRETZ2021_LONG, 'comment': "singular_singular_plural"},
        "lakretz2021_L_C": {'path': LAKRETZ2021_LONG, 'comment': "plural_plural_plural"},
        "lakretz2021_L_D": {'path': LAKRETZ2021_LONG, 'comment': "plural_plural_singular"},
    }

    datasets = list(dataset_dict.keys())
    checkpoints = list(MODEL_WEIGTHS.keys())

    combinations = product(datasets, checkpoints)

    # run every model on both datasets
    for data_checkpoint_tuple in combinations:

        dataset_key, checkpoint = data_checkpoint_tuple

        if checkpoint == "unablated":

            logging.info(f"Loading pretrained gpt2...")
            
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        elif checkpoint == "rand-init":

            logging.info(f"Loading randomly initialized gpt2...")

            config = GPT2LMHeadModel.from_pretrained("gpt2").config

            model = GPT2LMHeadModel(config)
            tokenizer =  GPT2TokenizerFast.from_pretrained("gpt2")

        else:
            
            logging.info(f"Loading {MODEL_WEIGTHS[checkpoint]}...")
            config = GPT2LMHeadModel.from_pretrained("gpt2").config
            model = GPT2LMHeadModel(config)
            model.load_state_dict(torch.load(MODEL_WEIGTHS[checkpoint]))

            tokenizer =  GPT2TokenizerFast.from_pretrained("gpt2")
        
        # ===== RUN EXPERIMENT ===== #
        logging.info(f"Running experiment for: {dataset_key} | combination: {dataset_dict[dataset_key]['comment']}")
        outputs = run_experiment(dataset_path=dataset_dict[dataset_key]['path'],
                                 model=model,
                                 tokenizer=tokenizer,
                                 comment=dataset_dict[dataset_key]['comment'],
                                 experiment_id=f"{checkpoint}_{dataset_key}"
                                )

        # save outputs
        savename = os.path.join(args.savedir, f"sva_{checkpoint}_{dataset_key}.json")
        logging.info(f"Saving {savename}")
        with open(savename, 'w') as fh:
            json.dump(outputs, fh, indent=4)


if __name__ == "__main__":

    main()
