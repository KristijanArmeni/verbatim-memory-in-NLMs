import os
import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Dict, Tuple
import numpy as np
import torch
from typing import List, Tuple, Dict
from tqdm import tqdm

# own modules
from eval_sets import sva_lakretz, sva_short_labels, sva_long_deps, sva_heterogenous
from paths import get_paths
from utils import logger
from wm_suite.wm_ablation import (
    ablate_attn_module,
    find_topk_attn,
    find_topk_intersection,
    get_pairs,
    from_dict_to_labels,
    from_labels_to_dict,
)


# define dicts that hold input arguments to create ablated models
all_heads = list(range(12))
single_heads = [[i] for i in list(range(12))]
single_layers = [[i] for i in list(range(12))]
layers_dict = {"".join([str(e) for e in a]): a for a in [[4, 5], [6, 7], [4, 5, 6, 7]]}


def read_json_data(path: str) -> Dict:
    logger.info(f"Loading {path}...")
    with open(path, "r") as fh:
        d = json.load(fh)

    return d["examples"]


def get_inputs_and_targets(examples_list: Dict, comment: str) -> Tuple:
    has_comment = False
    if "comment" in examples_list[0].keys():
        has_comment = True

    # lakretz dataset comes in 4 versions, make it possible to only use a subsest of versions by adding <comment>
    if has_comment & (comment != ""):
        inputs = [e["input"].strip() for e in examples_list if e["comment"] == comment]
        targets = [
            e["target_scores"] for e in examples_list if (e["comment"] == comment)
        ]
    else:
        # linzen dataset has whole sentences as targets, need to do .split() first
        inputs = [e["input"].strip() for e in examples_list]
        targets = [
            {
                key.split(" ")[0]: e["target_scores"][key]
                for key in e["target_scores"].keys()
            }
            for e in examples_list
            if (len(e["target_scores"]) == 2)
        ]

    assert len(inputs) == len(targets)
    logger.info(f"Number of examples == {len(inputs)}")

    return inputs, targets


def get_agreement_score(logits: torch.Tensor, targets: Tuple, tokenizer) -> int:
    probs = torch.softmax(logits, 0)  # convert logits to probabilities

    tar = list(targets.keys())

    # make sure we add space as a word marker to the string itself
    # otherwise, the BPE tokenizer assumes the string is attached to another strin and BPE splits it
    corr_id = tokenizer.encode(
        " " + tar[0]
    )  # the first item in the <tar> list is always the correct one
    incorr_id = tokenizer.encode(" " + tar[1])

    # if there's more than to IDs, it means the verb was BPE-split, skip it
    # there are not that many that need to be skipped
    if (len(corr_id) > 1) or (len(incorr_id) > 1):
        # logger.info("Skipping due to BPE split")
        out = {}

    else:
        # observed probabilities
        prob_correct = probs[corr_id].item()
        prob_incorrect = probs[incorr_id].item()

        agreement_correct = int(prob_correct > prob_incorrect)

        # print(strings)
        # print((prob_correct, prob_incorrect))
        strings = tuple(tokenizer.decode(id) for id in [corr_id, incorr_id])
        out = {key: v for key, v in zip(strings, (prob_correct, prob_incorrect))}

        out["res"] = agreement_correct
        out["max"] = torch.max(probs).cpu().item()
        out["corr_percentile"] = (torch.sum(probs > prob_correct).cpu().item()) / len(
            probs
        )

    return out


def collect_agreement_scores(
    model, inputs, targets, attention_mask_positions, tokenizer
):
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
        logger.info(
            f"Attention masking position {attention_mask_positions} in sequences"
        )

        attention_mask = torch.ones(size=(1, len(inputs[0])), device=model.device)
        attention_mask[0, torch.tensor(attention_mask_positions)] = 0

    for inp, tgt in tqdm(zip(inputs, targets), total=len(inputs)):
        o = model(input_ids=inp, attention_mask=attention_mask)

        s = get_agreement_score(o.logits[-1, :], targets=tgt, tokenizer=tokenizer)

        scores.append(s)

    return scores


def get_accuracy(scores: Dict) -> float:
    res = [
        e["res"] for e in scores if bool(e)
    ]  # if <e> is empty, it means the targets were BPE split, we skip these cases
    return round(sum(res) / len(res), 4)


def inputs2tensor(inputs, tokenizer):
    return torch.LongTensor([tokenizer.encode(x) for x in inputs])


def mask_inputs(inputs: torch.Tensor, masked_positions: List, masked_value: str):
    logger.info(f"Masking input as positions {masked_positions} with {masked_value}")
    out = []
    for i, inp in enumerate(inputs):
        tokens = inp.split(" ")
        for pos in masked_positions:
            tokens[pos] = masked_value

        out.append(" ".join(tokens))

    return out


def run_experiment(dataset_path, model, tokenizer, cfg, comment, experiment_id):
    baseline = cfg["baseline"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    examples = read_json_data(dataset_path)

    if comment:
        inputs, targets = get_inputs_and_targets(
            examples_list=examples, comment=comment
        )
    else:
        inputs, targets = get_inputs_and_targets(examples_list=examples, comment=None)

    logger.info("Collecting agreement scores...")
    logger.info(f"Example input: {inputs[5]}")
    scores = collect_agreement_scores(
        model.to(device),
        inputs=inputs2tensor(inputs, tokenizer).to(device),
        targets=targets,
        attention_mask_positions=None,
        tokenizer=tokenizer,
    )

    if baseline:
        logger.info("Collecting baseline agreement scores")

        mask_positions = [
            1,
            4,
        ]  # masking out the first and the second noun caryying grammatical number information
        if comment in [
            "singular_singular_singular",
            "plural_plural_plural",
            "singular_singular_plural",
            "plural_plural_singular",
        ]:
            mask_positions = [1, 4, 7]

    if cfg["input_masking"]:
        inputs = mask_inputs(
            inputs=inputs,
            masked_positions=mask_positions,
            masked_value=tokenizer.unk_token,
        )
        logger.info(f"Using masked inputs for baseline")
        logger.info(f"Example baseline input: {inputs[5]}")

    if baseline:
        baseline_scores = collect_agreement_scores(
            model.to(device),
            inputs=inputs2tensor(inputs, tokenizer).to(device),
            targets=targets,
            attention_mask_positions=mask_positions,
            tokenizer=tokenizer,
        )

    output = {}
    output["scores"] = scores
    output["accuracy"] = get_accuracy(scores)
    output["accuracy_baseline"] = get_accuracy(baseline_scores) if baseline else "none"
    output["accuracy_adj"] = (
        get_accuracy(scores) - get_accuracy(baseline_scores) if baseline else "none"
    )
    output["dataset"] = dataset_path
    output["type"] = comment
    output["id"] = experiment_id
    output["cfg"] = cfg

    logger.info("\nRESULTS")
    logger.info(
        f"\n{comment}:\naccuracy {output['accuracy']} | baseline {output['accuracy_baseline']} | adjusted {output['accuracy_adj']}\n"
    )

    return output


def get_test_input_args():
    logger.warning("Using test input_args, not collected through script!")

    input_args = [
        "--outname",
        "svatest.csv",
        "--savedir",
        "./",
        "--model_type",
        "ablated",
        "--model_id",
        "test-id",
        "--lh_dict",
        "{0: [0, 1, 2], 5: [5, 8, 9]}",
    ]

    return input_args


def main(input_args=None, devtesting=False):
    import argparse
    from ast import literal_eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--outname", type=str)
    parser.add_argument("--lh_dict", type=str)
    parser.add_argument(
        "--model_type", type=str, choices=["unablated", "rand-init", "ablated"]
    )
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--run_tag", type=str)
    parser.add_argument("--savedir", type=str)

    # this defaults to false if called as a script
    # set this varibale manually if doing interactive development
    if devtesting:
        input_args = get_test_input_args()
    print(input_args)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    # set seed for reproducibility
    torch.manual_seed(54321)

    # this log will go into a dataframe for compact storing of main results
    log = {
        "deplen": [],  # long vs. short
        "agr": [],  # agreement condition
        "model": [],  # unablated etc.
        "acc": [],  # raw accuracy
        "acc0_A": [],  # baseline accuracy (attention masking only)
        "tag": [],  # run tag
    }

    # we only evaluate on datasets where two nouns are of a different number (e.g. the <man> next to the <houses> is/are red)
    selected_keys = [
        e
        for e in list(sva_lakretz.keys())
        if sva_lakretz[e]["comment"] in sva_heterogenous
    ]

    # run every model on both datasets
    for dataset_key in selected_keys:
        ds_path = sva_lakretz[dataset_key]["path"]
        comment = sva_lakretz[dataset_key]["comment"]

        log["model"].append(args.model_id)
        log["agr"].append(sva_short_labels[comment])
        log["deplen"].append("long" if comment in sva_long_deps else "short")

        if args.model_type == "unablated":
            logger.info(f"Loading pretrained gpt2...")

            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        elif args.model_type == "rand-init":
            logger.info(f"Loading randomly initialized gpt2...")

            config = GPT2LMHeadModel.from_pretrained("gpt2").config

            model = GPT2LMHeadModel(config)
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        elif args.model_type == "ablated":
            # ablate the model
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model = ablate_attn_module(
                model, layer_head_dict=literal_eval(args.lh_dict), ablation_type="zero"
            )

            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # ===== RUN EXPERIMENT ===== #
        logger.info("\n==== STARTING EXPERIMENTS =====\n")

        logger.info(
            f"Running experiment for:\n"
            + f"dataset: {dataset_key}\n"
            + f"combination: {comment}\n"
            + f"model: {args.model_id}"
        )

        outputs1 = run_experiment(
            dataset_path=ds_path,
            model=model,
            tokenizer=tokenizer,
            cfg={"baseline": False, "input_masking": False},
            comment=comment,
            experiment_id=f"{args.model_id}_{dataset_key}",
        )

        log["acc"].append(outputs1["accuracy"])
        log["acc0_A"].append(outputs1["accuracy_baseline"])

        log["tag"].append(args.run_tag)

        # save outputs
        savename = os.path.join(args.savedir, f"sva_{args.model_id}_{dataset_key}.json")
        logger.info(f"Saving {savename}\n")
        with open(savename, "w") as fh:
            json.dump(outputs1, fh, indent=4)

        logger.info("\n===== DONE =====\n")

        df = pd.DataFrame(log)

        savename = os.path.join(args.savedir, args.outname)
        logger.info(f"Saving {savename}\n")
        df.to_csv(savename, sep="\t")

    logger.info("===== END OF RUN =====\n")


if __name__ == "__main__":
    main()
