import os
import json
import numpy as np
from typing import List
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(funcName)s() | %(message)s")

# own modules
from paths import PATHS
from wm_suite import wm_test_suite
from wm_suite.wm_ablation import from_labels_to_dict, from_dict_to_labels, find_topk_attn


def greedy_search(max_pair: List[str], remaining_heads:List[str], log_path: str, log_id:str) -> (List[float], List[str], dict, dict):
    """
    Parameters
    ----------
    max_pair : List[str]
        The pair of heads with the highest effect
    remaining_heads : List[str]
        The list of heads that are not in the max_pair
    
    Returns
    -------
    best_scores : List[float]
        The best scores for each step of the search
    best_labels : List[str]
        The best labels for each step of the search
    all_scores : dict
        The searched scores for each step of the search
    all_labels : dict
        The searched labels for each step of the search
    
    Examples
    --------
    ```python
    >>> max_pair = ["L0.H10", "L1.H11"]
    >>> remaining_heads = ["L3.H0", "L4.H7", "L4.H10", "L5.H11"]

    >>> best_scores, labels, searched_scores, searched_labels = greedy_search(max_pair, remaining_heads)
    ```
    """

    # start by ablating the pair with highest effect
    current_best = max_pair

    best_scores = []
    best_labels = []

    # store the whole search space for quality check
    all_scores = {}
    all_labels = {}
    patience = 0
    patience_limit = 2
    counter = 0

    # run the search until we run out of heads or if we have not improved for 2 steps 
    while len(remaining_heads) > 0 or patience < patience_limit:

        best_score = -1            # dummy score to start with
        best_combination = None    # dummy label
        scores = np.zeros(len(remaining_heads))
        head_labels = np.array(remaining_heads)
        searched_labels = []

        print("\n====================================")
        logging.info(f"Doing search (N_iterations = {len(remaining_heads)} | patience = {patience}/{patience_limit})")
        logging.info(f"Current best: {' '.join(current_best)}")
        logging.info(f"Current best score: {best_score}")
        print("====================================\n")

        for i, new_head in enumerate(remaining_heads):

            lh_dict = from_labels_to_dict(current_best + [new_head])

            input_args =  [
            "--scenario", "sce1",
            "--condition", "repeat",
            "--list_type", "random",
            "--list_len", "3",
            "--prompt_len", "1",
            "--model_type", "ablated",
            "--model_id", f"ablate_{i}_{new_head}",
            "--aggregate_output",
            "--aggregate_positions", "[0]",
            "--ablate_layer_head_dict", f"{lh_dict}",
            "--ablation_type", "zero",
            "--checkpoint", "gpt2",
            "--tokenizer", "gpt2",
            "--batch_size", "10",
            "--model_seed", "12345",
            "--noun_list_file", "/home/ka2773/project/lm-mem/src/data/noun_lists/random_lists.json",
            "--device", "cuda",
            ]

            # run the ablation experiment
            output = wm_test_suite.main(input_args)

            repsurp = np.round(output["median"], 2)

            # log the score
            scores[i] = np.round(repsurp, 2)

            # update the best score
            if repsurp > best_score:
                best_score = repsurp
                best_combination = current_best + [new_head]
        
            #print feedback
            print("\n====================================")
            logging.info("Current score: " + str(repsurp))
            logging.info("Current best score: " + str(best_score))
            logging.info("Current best combination: " + ' '.join(best_combination))
            print("====================================\n")
            
            # if log path is specified, save logs
            if log_path:
                searched_labels.append(" ".join(current_best + [new_head]))
                tmpfn = os.path.join(log_path, f"log_scores_{log_id+'_'}{counter:02d}.json")
                tmpdict = {"scores": scores.tolist(), 
                           "labels": searched_labels,
                           }
                with open(tmpfn, "w") as fh:
                    json.dump(tmpdict, fh, indent=4)

        # find the best head to add
        best_head = head_labels[np.argmax(scores)]
        
        # if current best score is smaller than the one from the previous round, increase patience
        if best_scores and (best_score < best_scores[-1]):
            patience += 1
        
        best_scores.append(best_score)

        # check that the best score was tracked correctly
        assert best_score == np.max(scores)

        # update the current best combination
        current_best = current_best + [best_head]
        best_labels.append(tuple(current_best))

        # if log path is specified, save logs
        if log_path is not None:
            tmpfn = os.path.join(log_path, f"log_best_scores{'_' + log_id}.json")
            tmpdict = {"scores": best_scores, "labels": best_labels}
            with open(tmpfn, "w") as fh:
                json.dump(tmpdict, fh, indent=4)

        # store all the scores in this run for quality check
        all_scores[i] = scores.tolist()
        all_labels[i] = remaining_heads
        print(remaining_heads)

        # remove the best head from the list of remaining heads
        remaining_heads.remove(best_head)

        counter += 1

    print("\n===== Done =====")
    logging.info(f"Found best score: {np.max(best_scores)} for combination: {' '.join(best_labels[np.argmax(best_scores)])}")

    return best_scores, best_labels, all_scores, all_labels


def main(input_args=None):
    
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--head_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.head_type == "matching":

        max_pair = ["L0.H10", "L1.H11"]
        fn = os.path.join(PATHS.root, "data", "topk_heads", "top_20_matching.json")
        with open(fn, "r") as fh:
            remaining_heads = json.load(fh)["lh_list"]
        remaining_heads.remove("L0.H10")
        remaining_heads.remove("L1.H11")
        remaining_heads = remaining_heads

    elif args.head_type == "postmatch":

        max_pair = ["L10.H11", "L10.H0"]
        fn = os.path.join(PATHS.root, "data", "topk_heads", "top_20_postmatch.json")
        with open(fn, "r") as fh:
            remaining_heads = json.load(fh)["lh_list"]
        remaining_heads.remove("L10.H11")
        remaining_heads.remove("L10.H0")

    elif args.head_type == "recent":
        
        max_pair = ["L3.H2", "L2.H3"]
        fn = os.path.join(PATHS.root, "data", "topk_heads", "top_20_recent.json")
        with open(fn, "r") as fh:
            remaining_heads = json.load(fh)["lh_list"]
        remaining_heads.remove("L3.H2")
        remaining_heads.remove("L2.H3")    

    # run the search
    outputs = greedy_search(max_pair, remaining_heads, log_path=args.output_dir, log_id=args.head_type)
    
    outs = {
        "best_scores": outputs[0],
        "best_labels": outputs[1],
        "all_scores": outputs[2],
        "all_labels": outputs[3],
    }

    # save the outputs
    logging.info(f"Saving outputs to {os.path.join(args.output_dir, args.output_filename)}")
    with open(os.path.join(args.output_dir, args.output_filename), "w") as fh:
        json.dump(outs, fh, indent=4)


    return None


if __name__ == "__main__":
    
    main()


