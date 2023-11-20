
import json, os
import logging
import numpy as np
from paths import PATHS
from wm_suite.wm_ablation import find_topk_attn, from_dict_to_labels


cfg = {

    "p1": {
        "attn_w": PATHS.attn_w,
        "matching_toi": [13],
        "postmatch_toi": [14, 16, 18],
        "recent_toi": [44, 43, 42],
        "attn_threshold": 0,
    },
    # topk computed over single token
    "n1": {
        "attn_w": PATHS.attn_w_n1,
        "matching_toi": [14],
        "postmatch_toi": [16, 18],
        "recent_toi": [45, 44, 43],
        "attn_threshold": 0,
    },

    "p1_t1": {
        "attn_w": PATHS.attn_w,
        "matching_toi": [13],
        "postmatch_toi": [14],
        "recent_toi": [44],
        "attn_threshold": 0.2,
    }
}

save_joint = False

def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=str, default="p1")
    args = parser.parse_args()

    config = cfg[args.query_id]

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # load the weights
    attn = dict(np.load(config["attn_w"]))["data"]

    for k in [20]:
         
        print(k)
        # MATCHING HEADS
        toi = config["matching_toi"]
        lh_dict, _, _ = find_topk_attn(attn, topk=k, attn_threshold=config['attn_threshold'], tokens_of_interest=toi, seed=12345)
        lh_list = from_dict_to_labels(lh_dict)

        outdict = {
            "topk": k,
            "toi": toi,
            "lh_list": lh_list,
            "lh_dict": lh_dict,
        }

        fn = os.path.join(PATHS.root, "data", "topk_heads", f"top_{k:02d}_attn{config['attn_threshold']:.1f}_matching_{args.query_id}.json")
        logging.info(f"Saving {fn}")
        with open(fn, "w") as fh:
            json.dump(outdict, fh, indent=4)

        # POSTMATCH HEADS
        toi = config["postmatch_toi"]
        lh_dict, _, _ = find_topk_attn(attn, topk=k, attn_threshold=config['attn_threshold'], tokens_of_interest=toi, seed=12345)
        lh_list = from_dict_to_labels(lh_dict)

        outdict = {
            "topk": k,
            "toi": toi,
            "lh_list": lh_list,
            "lh_dict": lh_dict,
        }

        fn = os.path.join(PATHS.root, "data", "topk_heads", f"top_{k:02d}_attn{config['attn_threshold']:.1f}_postmatch_{args.query_id}.json")
        logging.info(f"Saving {fn}")
        with open(fn, "w") as fh:
            json.dump(outdict, fh, indent=4)

        # RECENT-TOKENS HEADS
        toi = config["recent_toi"]
        lh_dict, _, _ = find_topk_attn(attn, topk=k, attn_threshold=config['attn_threshold'], tokens_of_interest=toi, seed=12345)
        lh_list = from_dict_to_labels(lh_dict)

        outdict = {
            "topk": k,
            "toi": toi,
            "lh_list": lh_list,
            "lh_dict": lh_dict,
        }

        fn = os.path.join(PATHS.root, "data", "topk_heads", f"top_{k:02d}_attn{config['attn_threshold']:.1f}_recent_{args.query_id}.json")
        logging.info(f"Saving {fn}")
        with open(fn, "w") as fh:
            json.dump(outdict, fh, indent=4)

    return None


if __name__ == "__main__":
    main()

