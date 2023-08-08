
import json, os
import logging
import numpy as np
from paths import PATHS
from wm_suite.wm_ablation import find_topk_attn, from_dict_to_labels


def main():

    # load the weights
    attn = dict(np.load(PATHS.attn_w))["data"]

    for k in [5, 10, 15, 20]:
         
        print(k)
        # MATCHING HEADS
        toi = [13]
        lh_dict, _, _ = find_topk_attn(attn, topk=k, tokens_of_interest=toi, seed=12345)
        lh_list = from_dict_to_labels(lh_dict)

        outdict = {
            "topk": k,
            "toi": toi,
            "lh_list": lh_list,
            "lh_dict": lh_dict,
        }

        fn = os.path.join(PATHS.root, "data", "topk_heads", f"top_{k:02d}_matching.json")
        logging.info(f"Saving {fn}")
        with open(fn, "w") as fh:
            json.dump(outdict, fh, indent=4)

        # POSTMATCH HEADS
        toi = [14, 16, 18]
        lh_dict, _, _ = find_topk_attn(attn, topk=k, tokens_of_interest=toi, seed=12345)
        lh_list = from_dict_to_labels(lh_dict)

        outdict = {
            "topk": k,
            "toi": toi,
            "lh_list": lh_list,
            "lh_dict": lh_dict,
        }

        fn = os.path.join(PATHS.root, "data", "topk_heads", f"top_{k:02d}_postmatch.json")
        logging.info(f"Saving {fn}")
        with open(fn, "w") as fh:
            json.dump(outdict, fh, indent=4)

        # RECENT-TOKENS HEADS
        toi = [44, 43, 42]
        lh_dict, _, _ = find_topk_attn(attn, topk=k, tokens_of_interest=toi, seed=12345)
        lh_list = from_dict_to_labels(lh_dict)

        outdict = {
            "topk": k,
            "toi": toi,
            "lh_list": lh_list,
            "lh_dict": lh_dict,
        }

        fn = os.path.join(PATHS.root, "data", "topk_heads", f"top_{k:02d}_recent.json")
        logging.info(f"Saving {fn}")
        with open(fn, "w") as fh:
            json.dump(outdict, fh, indent=4)

    return None


if __name__ == "__main__":
    main()

