
from wm_suite.wm_test_suite import runtime_code
import argparse
import os
import logging

def test_fun(inps):

    parser = argparse.ArgumentParser()

    parser.add_argument("--arg1")
    parser.add_argument("--arg2")

    if inps:
        args = parser.parse_args(inps)
    else:
        args = parser.parse_args()
    
    return args

def reproduce_set_size_experiment():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()

    logging.info("Call to reproduce_set_size_experiment()")

    DATAPATH = "/home/ka2773/project/lm-mem/src/data/transformer_input_files/gpt2"

    fn = "gpt2_repeat_sce1_1_n3_categorized.json"

    input_arguments = [
        "--scenario", "sce1",
        "--condition", "repeat",
        "--inputs_file", os.path.join(DATAPATH, fn),
        "--inputs_file_info", os.path.join(DATAPATH, fn.replace(".json", "_info.json")),
        "--tokenizer", "gpt2",
        "--checkpoint", "gpt2",
        "--device", f"{args.device}",
        "--batch_size", f"{args.bs}",
        "--output_dir", "./",
        "--output_filename", "wms_gpt2_a-10_sce1_1_n3_repeat_categorized.csv"
    ]

    runtime_code(input_arguments)

    return 0