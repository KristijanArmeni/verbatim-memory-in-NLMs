import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import pandas as pd

from wm_suite.wm_test_suite import Experiment
from wm_suite.paths import add_data_to_syspath, get_paths
from wm_suite.utils import logger
from wm_suite.wm_ablation import ablate_attn_module, from_dict_to_labels

add_data_to_syspath()
from wt103.dataset import WikiTextDataset


def gpt2_wt103(model, tokenizer, context_len, stride, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = get_paths()

    logger.info(f"Loading {paths.wt103_test}...")
    _, ids = WikiTextDataset(tokenizer=tokenizer).retokenize_txt(paths.wt103_test)

    # initialize experiment class
    exp = Experiment(
        model=model,
        ismlm=False,
        tokenizer=tokenizer,
        context_len=context_len,
        batch_size=batch_size,
        stride=stride,
        use_cache=False,
        device=device,
    )

    # compute Wikitext-103 perplexity
    ppl, _, _ = exp.ppl(
        input_ids=torch.tensor([ids]), context_len=context_len, stride=stride
    )

    return ppl


def main(input_args=None):
    import argparse
    from ast import literal_eval

    parser = argparse.ArgumentParser(
        description="Compute Wikitext-103 perplexity for different ablations of the GPT-2 model."
    )

    parser.add_argument("--context_len", type=int, default=1024, help="context length")
    parser.add_argument("--stride", type=int, default=256, help="stride")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["unablated", "ablated", "rand-init"],
        default="unablated",
        help="model type",
    )
    parser.add_argument(
        "--lh_dict", type=str, default=None, help="layer-head dictionary"
    )
    parser.add_argument("--savename", type=str, default=None, help="save name")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if args.model_type == "ablated":
        assert (
            args.lh_dict is not None
        ), "Please provide a layer-head dictionary (--lh_dict)"

        lh_dict = literal_eval(args.lh_dict)

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = ablate_attn_module(model, layer_head_dict=lh_dict, ablation_type="zero")

        label = "_".join(from_dict_to_labels(lh_dict))

    elif args.model_type == "rand-init":
        config = GPT2LMHeadModel.from_pretrained("gpt2").config
        model = GPT2LMHeadModel(config=config)

        label = args.model_type
        lh_dict = {}  # just for the step value in the output dict

    elif args.model_type == "unablated":
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        label = args.model_type
        lh_dict = {}

    ppl = gpt2_wt103(
        model=model,
        tokenizer=tokenizer,
        context_len=args.context_len,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    out = pd.DataFrame(
        {
            "ppl": ppl.cpu().item(),
            "lh_dict": label,
            "step": len(from_dict_to_labels(lh_dict)),
        },
        index=[0],
    )

    if args.savename is not None:
        out.to_csv(args.savename, sep="\t")

    return out


if __name__ == "__main__":
    main()
