
import os
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Dict, Tuple
import torch
import logging
from typing import List, Tuple, Dict
from tqdm import tqdm
from itertools import product
from wm_suite.wm_ablation import ablate_attn_module

logging.basicConfig(level=logging.INFO, format="%(message)s")

LINZEN2016 = "/scratch/ka2773/project/lm-mem/sv_agr/linzen2016/linzen2016_english.json"
LAKRETZ2021_SHORT = "/scratch/ka2773/project/lm-mem/sv_agr/lakretz2021/short_nested_outer_english.json"
LAKRETZ2021_LONG = "/scratch/ka2773/project/lm-mem/sv_agr/lakretz2021/long_nested_outer_english.json"

# define dicts that hold input arguments to create ablated models
all_heads = list(range(12))
single_layers = [[i] for i in list(range(12))]
layers_dict = {"".join([str(e) for e in a]): a for a in single_layers + [[0,1], [2, 3], [0, 1, 2, 3], [8, 9], [10, 11], [8, 9, 10, 11]]}

ablation_dict = {f"ablate-{key}": {"layers": layers_dict[key], "heads": all_heads} for key in layers_dict}
ablation_dict["ablate-all"] = {'layers': list(range(12)), "heads": all_heads}
ablation_dict["rand-init"] = "rand-init"
ablation_dict["unablated"] = "gpt2"


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
    corr_id = tokenizer.encode(" " + tar[0])    # the first item in the <tar> list is always the correct one
    incorr_id = tokenizer.encode(" " + tar[1])

    # if there's more than to IDs, it means the verb was BPE-split, skip it
    # there are not that many that need to be skipped
    if (len(corr_id) > 1) or (len(incorr_id) > 1):
        #logging.info("Skipping due to BPE split")
        out = {}

    else:

        # observed probabilities
        prob_correct = probs[corr_id].item()
        prob_incorrect = probs[incorr_id].item()

        agreement_correct = int(prob_correct > prob_incorrect)

        #print(strings)
        #print((prob_correct, prob_incorrect))
        strings = tuple(tokenizer.decode(id) for id in [corr_id, incorr_id])
        out = {key: v for key, v in zip(strings, (prob_correct, prob_incorrect))}

        out['res'] = agreement_correct
        out['max'] = torch.max(probs).cpu().item()
        out['corr_percentile'] = (torch.sum(probs > prob_correct).cpu().item())/len(probs)

    return out


def collect_agreement_scores(model, inputs, targets, attention_mask_positions, tokenizer):
    """
    Parameters:
    ----------
    model : transformers.PreTrainedTransformer
    inputs : torch.tensor, shape = (n_samples, sequence_len)
    """

    scores = []
    
    # set attention masking if requested
    attention_mask = None
    if attention_mask_positions is not None:

        logging.info(f"Attention masking position {attention_mask_positions} in sequences")
        
        attention_mask = torch.ones(size=(1, len(inputs[0])), device=model.device)
        attention_mask[0, torch.tensor(attention_mask_positions)] = 0
        
    for inp, tgt in tqdm(zip(inputs, targets), total=len(inputs)):

        o = model(input_ids=inp, attention_mask=attention_mask)

        s = get_agreement_score(o.logits[-1, :], targets=tgt, tokenizer=tokenizer)

        scores.append(s)

    return scores


def get_accuracy(scores: Dict) -> float:

    res = [e['res'] for e in scores if bool(e)]  # if <e> is empty, it means the targets were BPE split, we skip these cases
    return round(sum(res)/len(res), 4)


def inputs2tensor(inputs, tokenizer):
    return torch.LongTensor([tokenizer.encode(x) for x in inputs])


def mask_inputs(inputs: torch.Tensor, masked_positions: List, masked_value: str):
    
    logging.info(f"Masking input as positions {masked_positions} with {masked_value}")
    out = []
    for i, inp in enumerate(inputs):
        tokens = inp.split(" ")
        for pos in masked_positions:
            tokens[pos] = masked_value

        out.append(" ".join(tokens))

    return out


def run_experiment(dataset_path, model, tokenizer, cfg, comment, experiment_id):


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    examples = read_json_data(dataset_path)

    if comment:
        inputs, targets = get_inputs_and_targets(examples_list=examples, comment=comment)
    else:
        inputs, targets = get_inputs_and_targets(examples_list=examples, comment=None)

    logging.info("Collecting agreement scores...")
    logging.info(f"Example input: {inputs[5]}")
    scores = collect_agreement_scores(model.to(device), 
                                      inputs=inputs2tensor(inputs, tokenizer).to(device), 
                                      targets=targets,
                                      attention_mask_positions=None,
                                      tokenizer=tokenizer)


    logging.info("Collecting baseline agreement scores")
        
    mask_positions = [1, 4]  # masking out the first and the second noun caryying grammatical number information
    if comment in ["singular_singular_singular", "plural_plural_plural", "singular_singular_plural", "plural_plural_singular"]:
        mask_positions = [1, 4, 7]


    if cfg['input_masking']:

        inputs = mask_inputs(inputs=inputs, 
                             masked_positions=mask_positions,
                             masked_value=tokenizer.unk_token)
        logging.info(f"Using masked inputs for baseline")
        logging.info(f"Example baseline input: {inputs[5]}")


    baseline_scores = collect_agreement_scores(model.to(device), 
                                               inputs=inputs2tensor(inputs, tokenizer).to(device), 
                                               targets=targets,
                                               attention_mask_positions=mask_positions,
                                               tokenizer=tokenizer)

    output = {}
    output["scores"] = scores
    output["accuracy"] = get_accuracy(scores)
    output["accuracy_baseline"] = get_accuracy(baseline_scores)
    output["accuracy_adj"] = get_accuracy(scores) - get_accuracy(baseline_scores)
    output["dataset"] = dataset_path
    output["type"] = comment
    output['id'] = experiment_id
    output["cfg"] = cfg

    logging.info("\nRESULTS")
    logging.info(f"\n{comment}:\naccuracy {output['accuracy']} | baseline {output['accuracy_baseline']} | adjusted {output['accuracy_adj']}\n")
    

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
        #"linzen2016": {'path': LINZEN2016, 'comment': None}, 
        "lakretz2021_S_A": {'path': LAKRETZ2021_SHORT, 'comment': "singular_singular"},
        "lakretz2021_S_B": {'path': LAKRETZ2021_SHORT, 'comment': "singular_plural"},
        "lakretz2021_S_C": {'path': LAKRETZ2021_SHORT, 'comment': "plural_singular"},
        "lakretz2021_S_D": {'path': LAKRETZ2021_SHORT, 'comment': "plural_plural"},
        "lakretz2021_L_A": {'path': LAKRETZ2021_LONG, 'comment': "singular_singular_singular"},
        "lakretz2021_L_B": {'path': LAKRETZ2021_LONG, 'comment': "singular_singular_plural"},
        "lakretz2021_L_C": {'path': LAKRETZ2021_LONG, 'comment': "plural_plural_plural"},
        "lakretz2021_L_D": {'path': LAKRETZ2021_LONG, 'comment': "plural_plural_singular"},
    }

    long_deps = ["singular_singular_singular", "singular_singular_plural", "plural_plural_plural", "plural_plural_singular"]
    short_labels = {
        "singular_singular": "SS",
        "singular_plural": "SP",
        "plural_singular": "PS",
        "plural_plural": "PP",
        "singular_singular_singular": "SSS",
        "singular_singular_plural": "SSP",
        "plural_plural_plural": "PPP",
        "plural_plural_singular": "PPS",
    }

    datasets = list(dataset_dict.keys())
    model_names = list(ablation_dict.keys())

    combinations = list(product(datasets, model_names))

    # this log will go into a dataframe for compact storing of main results
    log = {
        "deplen": [],     # long vs. short
        "agr": [],        # agreement condition
        "model": [],      # unablated etc.
        "acc": [],        # raw accuracy
        "acc0_A": [],     # baseline accuracy (attention masking only)
        "acc0_B": [],     # baseline accuracy (attention and input masking)
    }

    # run every model on both datasets
    for data_model_name_tuple in combinations:

        dataset_key, model_name = data_model_name_tuple

        ds_path = dataset_dict[dataset_key]['path']
        comment = dataset_dict[dataset_key]['comment']

        log['model'].append(model_name)
        log['agr'].append(short_labels[comment])
        log['deplen'].append("long" if comment in long_deps else "short")

        if model_name == "unablated":

            logging.info(f"Loading pretrained gpt2...")
            
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        elif model_name == "rand-init":

            logging.info(f"Loading randomly initialized gpt2...")

            config = GPT2LMHeadModel.from_pretrained("gpt2").config

            model = GPT2LMHeadModel(config)
            tokenizer =  GPT2TokenizerFast.from_pretrained("gpt2")

        else:

            # ablate the model
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model = ablate_attn_module(model, 
                                       layers=ablation_dict[model_name]['layers'],
                                       heads=ablation_dict[model_name]['heads'],
                                       ablation_type="zero")

            tokenizer =  GPT2TokenizerFast.from_pretrained("gpt2")
        

        # ===== RUN EXPERIMENT ===== #
        logging.info("\n==== STARTING EXPERIMENTS =====\n")

        logging.info(f"Running experiment for:\n" +
                     f"dataset: {dataset_key}\n" + 
                     f"combination: {comment}\n" +
                     f"model: {model_name}")

        outputs1 = run_experiment(dataset_path=ds_path,
                                 model=model,
                                 tokenizer=tokenizer,
                                 cfg={"input_masking": False},
                                 comment=comment,
                                 experiment_id=f"{model_name}_{dataset_key}"
                                )

        log['acc'].append(outputs1['accuracy'])
        log['acc0_A'].append(outputs1['accuracy_baseline'])
   
        # save outputs
        savename = os.path.join(args.savedir, f"sva_{model_name}_{dataset_key}.json")
        logging.info(f"Saving {savename}\n")
        #with open(savename, 'w') as fh:
        #    json.dump(outputs1, fh, indent=4)

        logging.info(f"Running input masking experiment for: {dataset_key} | combination: {dataset_dict[dataset_key]['comment']}")
        outputs2 = run_experiment(dataset_path=ds_path,
                                 model=model,
                                 tokenizer=tokenizer,
                                 cfg={"input_masking": True},
                                 comment=comment,
                                 experiment_id=f"{model_name}_{dataset_key}"
                                )

        log['acc0_B'].append(outputs2['accuracy_baseline'])

        # save outputs
        savename = os.path.join(args.savedir, f"sva_{model_name}_{dataset_key}_input-masking.json")
        logging.info(f"Saving {savename}\n")
        #with open(savename, 'w') as fh:
        #    json.dump(outputs2, fh, indent=4)

        logging.info("\n===== DONE =====\n")

    df = pd.DataFrame(log)

    savename = os.path.join(args.savedir, f"ablation_sva.csv")
    logging.info(f"Saving {savename}\n")
    df.to_csv(savename, sep="\t")

    logging.info("===== END OF RUN =====\n")

if __name__ == "__main__":

    main()
