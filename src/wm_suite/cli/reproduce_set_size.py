from itertools import product
from wm_suite.wm_test_suite import runtime_code, make_filename_from_config
from wm_suite.utils import DATAPATH

import argparse
import os
import logging


def reproduce_set_size_experiment():
    """
    reproduce_set-size_experiment() is a CLI wrapper around wm_test_suite.runtime_code()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    print("=====")
    logging.info("Call to reproduce_set_size_experiment()")
    logging.info(f"Looking for data in: {DATAPATH}")
    logging.info(f"Will be saving to: {os.path.abspath(args.output_dir)}")
    print("=====")

    vignettes = ["sce1"]
    conditions = ["repeat"]
    list_lengths = [3, 5, 7, 10]
    intervening_texts = [1]
    list_types = ["random"]

    # get all combinations of input arguments
    runs = list(
        product(vignettes, conditions, list_lengths, list_types, intervening_texts)
    )

    # loop over all combinations
    for run in runs:
        # expand the tuple for the current combination
        vignette, condition, list_len, list_type, intervening_text = run

        fn, fn2 = make_filename_from_config(
            model=args.model,
            vignette=vignette,
            condition=condition,
            list_len=list_len,
            list_type=list_type,
            prompt=intervening_text,
        )

        input_arguments = [
            "--scenario",
            vignette,
            "--condition",
            condition,
            "--inputs_file",
            os.path.join(DATAPATH, fn),
            "--inputs_file_info",
            os.path.join(DATAPATH, fn2),
            "--tokenizer",
            args.model,
            "--checkpoint",
            args.model,
            "--device",
            f"{args.device}",
            "--batch_size",
            f"{args.bs}",
            "--output_dir",
            f"{os.path.abspath(args.output_dir)}",
            "--output_filename",
            f"wms_gpt2_a-10_{vignette}_{intervening_text}_n{list_len}_{condition}_{list_type}.csv",
        ]

        runtime_code(input_arguments)

    return 0
