import os
import json
import numpy as np
from typing import List

# own modules
from .utils import logger
from .paths import get_paths
from . import wm_test_suite
from .wm_ablation import from_labels_to_dict


class Criterion():

    def __init__(self, patience_limit, max_iter):

        self.patience_limit = patience_limit
        self.max_iter = max_iter


    def check(self, search_space, patience, counter):

        # if max iter is not set
        # check that the patience limit is not reached and the remaining search space is not empty
        condition1 = len(search_space) > 0 and (patience < self.patience_limit)
        
        # if max iteration limit is set, include it in the stopping criterion
        if self.max_iter:    
            condition2 = self.max_iter > counter
            stopnow = condition1 and condition2
        else:
            stopnow = condition1

        return stopnow


def greedy_search(attention_heads: List[str],
                  repeat_surprisal_timesteps: List[int], 
                  list_len: int,
                  save_every: int,
                  log_path: str,
                  log_id: str,
                  max_iter: int = None,
                  patience_limit: int = 0) -> (List[float], List[str], dict, dict):
    """
    Parameters
    ----------
    attention_heads : List[str]
        The list of heads that are not in the max_pair
    repeat_surprisal_timesteps : List[int]
        The timesteps for which to compute the repeat surprisal
    list_len : int
        The number of nouns in the lists
    save_every : int
        How often to save the outputs
    log_path : str
        The path to the log directory
    log_id : str
        The id of the log, will used to construct the log filenames
    max_iter : int
        The maximum number of iterations to run the search for (if patience limit is not yet reached)
    patience_limit : int
        value idicating the number iterations to perform once the search has stopped increasing by > 0.5% repeat surprisal
    
    Returns
    -------
    Dict :
        The outputs of the search stored in a dict with keys:
        - rs : repeat surprisal scores (and 95% ci) for each step of the search
        - x1 : raw surprisal on first list
        - x2 : raw surprisal on second list
        - best_labels : the best layer/head label for each step of the search
    
    Examples
    --------
    ```python
    >>> attention_heads = ["L3.H0", "L4.H7", "L4.H10", "L5.H11"]

    >>> output_dict = greedy_search(attention_heads)
    ```
    """

    n_total_runs = len(attention_heads)

    # start by ablating the pair with highest effect
    current_best = []

    best_scores = []                                # for storing repeat surprisal scores
    best_scores_ci = []                             # for storing confidence intervals
    best_scores_x1, best_scores_x2 = [], []         # for storing raw surprisal values
    best_scores_x1_ci, best_scores_x2_ci = [], []   # for storing raw surprisal values
    best_labels = []                                # for storing the labels for each step of the search
    # store the whole search space for quality check
    all_scores = {}
    all_labels = {}
    patience = 0
    counter = 0

    if patience_limit == 0:
        patience_limit = n_total_runs  # if not specify, set patience limit to the number of total runs
        logger.info(f"Setting patience limit to N_total {patience_limit} iterations")

    # function for saving outputs every `save_every` iterations
    def create_outputs(rs, rs_ci, rs_labels, x1, x1_ci, x2, x2_ci, all_rs, all_labels):

        outs = {
            "rs": {"scores": rs, "ci": rs_ci },
            "x1": {"scores": x1, "ci": x1_ci},
            "x2": {"scores": x2, "ci": x2_ci},
            "best_labels": rs_labels,
            "searched_scores": all_rs,
            "searched_labels": all_labels,
        }

        return outs


    early_stop = Criterion(patience_limit=patience_limit, max_iter=max_iter)

    # run the search until we run out of heads or if we have not improved for 2 rounds 
    while early_stop.check(search_space=attention_heads, patience=patience, counter=counter):

        best_score = -1            # dummy score to start with
        best_combination = None    # dummy label
        scores = np.zeros(len(attention_heads))
        head_labels = np.array(attention_heads)
        searched_labels = []

        print("\n====================================")
        logger.info(f"Doing search (N_iterations = {len(attention_heads)} | patience = {patience}/{patience_limit} | max_iter = {max_iter} | Save = every {save_every} iterations)")
        logger.info(f"Current best: {' '.join(current_best)}")
        logger.info(f"Current best score: {best_score}")
        print("====================================\n")

        for i, new_head in enumerate(attention_heads):

            lh_dict = from_labels_to_dict(current_best + [new_head])

            input_args =  [
            "--scenario", "sce1",
            "--condition", "repeat",
            "--list_type", "random",
            "--list_len", f"{list_len}",
            "--prompt_len", "1",
            "--model_type", "ablated",
            "--model_id", f"ablate_{i}_{new_head}",
            "--aggregate_output",
            "--aggregate_positions", f"{repeat_surprisal_timesteps}",
            "--ablate_layer_head_dict", f"{lh_dict}",
            "--ablation_type", "zero",
            "--checkpoint", "gpt2",
            "--tokenizer", "gpt2",
            "--batch_size", "23",
            "--model_seed", "12345",
            "--noun_list_file", "/home/ka2773/project/lm-mem/src/data/noun_lists/random_lists.json",
            "--device", "cuda",
            ]

            # run the ablation experiment
            output = wm_test_suite.main(input_args)

            # get values from the aggregated outputs
            repsurp = np.round(output["rs"]["median"], 2)
            repsurp_ci = output["rs"]["ci95"]
            x1 = np.round(output["x1"]["median"], 2)
            x2 = np.round(output["x2"]["median"], 2)
            x1_ci = output["x1"]["ci95"]
            x2_ci = output["x2"]["ci95"]

            # log the score
            scores[i] = repsurp

            # update the best score
            if repsurp > best_score:
                
                best_score = repsurp           # repeat surprisal
                best_score_ci = repsurp_ci     # repeat surprisal confidence interval
                best_score_x1 = x1             # raw surprisal on first list
                best_score_x2 = x2             # raw surprisal on second list
                best_score_x1_ci = x1_ci       # ci for raw surprisal on first list
                best_score_x2_ci = x2_ci       # ci for raw surprisal on second list
                best_combination = current_best + [new_head]
        
            #print feedback
            print("\n====================================")
            logger.info("Current score: " + str(repsurp))
            logger.info("Current best score: " + str(best_score))
            logger.info("Current best combination: " + ' '.join(best_combination))
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
        
        # if increment from past iteraion is smaller than 1%, increase patience
        increment = best_score - best_scores[-1] if best_scores else 1  # dummy value to keep the code running
        patience_delta = 0.5
        if increment < patience_delta:
            patience += 1
        # reset patience counter if increment is larger than 1% (growing again)
        elif (increment >= patience_delta) and (patience > 0):
            patience = 0
        
        # store the best score and its confidence interval and raw surprisals
        best_scores.append(best_score)
        best_scores_ci.append(best_score_ci)
        best_scores_x1.append(best_score_x1)
        best_scores_x2.append(best_score_x2)
        best_scores_x1_ci.append(best_score_x1_ci)
        best_scores_x2_ci.append(best_score_x2_ci)

        # check that the best score was tracked correctly
        assert best_score == np.max(scores)

        # update the current best combination
        current_best = current_best + [best_head]
        best_labels.append(tuple(current_best))

        # if log path is specified, save logs
        if log_path is not None:
            tmpfn = os.path.join(log_path, f"log_best_scores{'_' + log_id}.json")
            tmpdict = {"scores": best_scores, "scores_ci": best_scores_ci, "labels": best_labels}
            with open(tmpfn, "w") as fh:
                json.dump(tmpdict, fh, indent=4)

        # check if save_every is reached and save full output at current iteration
        if (counter+1) % save_every == 0:
            tmpfn = os.path.join(log_path, f"scores_{log_id}_iter-{counter+1:02d}.json")
            logger.info(f"Saving scores to {tmpfn}")
            outs = create_outputs(rs=best_scores, 
                                  rs_ci=best_scores_ci,
                                  rs_labels=best_labels,
                                  x1=best_scores_x1,
                                  x1_ci=best_scores_x1_ci,
                                  x2=best_scores_x2,
                                  x2_ci=best_scores_x2_ci,
                                  all_rs=all_scores,
                                  all_labels=all_labels)
            with open(tmpfn, "w") as fh:
                json.dump(outs, fh, indent=4)

        # store all the scores in this run for quality check
        all_scores[i] = scores.tolist()
        all_labels[i] = attention_heads

        # remove the best head from the list of remaining heads
        attention_heads.remove(best_head)

        counter += 1

    print("\n===== Done =====")
    if patience == patience_limit:
        logger.info(f"Reached patience limit of {patience_limit} iterations")
    logger.info(f"Iterations ran: {len(best_scores)}/{n_total_runs} | patience: {patience}/{patience_limit}")
    logger.info(f"Found best score: {np.max(best_scores)} for combination: {' '.join(best_labels[np.argmax(best_scores)])}")

    # create the final outputs for return
    outs = create_outputs(rs=best_scores, 
                          rs_ci=best_scores_ci,
                          rs_labels=best_labels,
                          x1=best_scores_x1,
                          x1_ci=best_scores_x1_ci,
                          x2=best_scores_x2,
                          x2_ci=best_scores_x2_ci,
                          all_rs=all_scores,
                          all_labels=all_labels)

    return outs


def get_input_args_for_devtesting():

    out = ["--lh_dict_json", "/home/ka2773/project/lm-mem/data/topk_heads/all_heads.json",
           "--log_id", "test",
           "--repeat_surprisal_timesteps", "[1,2]",
           "--list_len", "3",
           "--patience_limit", "5",
           "--output_dir", "./",
           "--output_filename", "test.json"]
    
    return out

def main(input_args=None, devtesting=False):
    
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--lh_dict_json", type=str, required=True)
    parser.add_argument("--log_id", type=str, required=True)
    parser.add_argument("--repeat_surprisal_timesteps", type=str, required=True)
    parser.add_argument("--list_len", type=int, required=True, default=3)
    parser.add_argument("--patience_limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)

    if devtesting:
        logger.warning("Running in devtesting mode!")
        input_args = get_input_args_for_devtesting()

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    logger.info("Using the following input arguments:\n" + "\n".join([f"{k}: {v}" for k, v in vars(args).items()]))

    # load in the top-k attention heads loaded aso
    with open(args.lh_dict_json, "r") as fh:
        attention_heads = json.load(fh)["lh_list"]

    # run the search
    outputs = greedy_search(attention_heads=attention_heads, 
                            repeat_surprisal_timesteps=args.repeat_surprisal_timesteps,
                            list_len=args.list_len,
                            save_every=5,
                            patience_limit=args.patience_limit,
                            log_path=args.output_dir,
                            log_id=args.log_id)

    # save the outputs
    logger.info(f"Saving outputs to {os.path.join(args.output_dir, args.output_filename)}")
    with open(os.path.join(args.output_dir, args.output_filename), "w") as fh:
        json.dump(outputs, fh, indent=4)


    return None


if __name__ == "__main__":
    
    main()


