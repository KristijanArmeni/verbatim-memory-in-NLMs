# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:03:44 2021

"""

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import argparse
import os
import json
import logging

logging.basicConfig(format="INFO: %(message)s", level=logging.INFO)


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer):
        self.x = None  # input samples
        self.tokenizer = tokenizer

    # split dataset into chunks of 1024 tokens
    def chunk_list(self, l, n, equal_len=True, bos=None, eos=None):
        n = max(1, n)

        if (bos is not None) and (eos is not None):
            # if bos and eos need to be added adjust selected n to not exceed
            # the allowed maximum
            logging.info("Adding bos and eos setting {} to {}".format(n, n - 2))
            n -= 2

        if equal_len:
            total_len = (len(l) // n) * n
        else:
            total_len = len(l)

        logging.info("Creating {} input sequences of length {}".format(total_len, n))
        output_list = (
            [bos] + l[i : i + n] + [eos] for i in tqdm(range(0, total_len, n))
        )

        return output_list

    def retokenize_txt(self, path, save_retokenized=False):
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # retokenize the entire test set
        logging.info("Retokenizing {}".format(path))

        tokens = []
        ids = []

        for l in tqdm(lines, desc="line"):
            tokens += self.tokenizer.tokenize(l)
            ids += self.tokenizer.encode(l)

        if save_retokenized:
            # save tokens as .json
            logging.info("Saving {}".format(save_retokenized))
            with open(save_retokenized, "w") as fname:
                json.dump(tokens, fname)

            # save indices as .json and make sure it ends with inds.bpe.json
            logging.info(
                "Saving {}".format(
                    save_retokenized.replace(".bpe.json", ".inds.bpe.json")
                )
            )
            with open(
                save_retokenized.replace(".bpe.json", ".inds.bpe.json"), "w"
            ) as fname:
                json.dump(ids, fname)

        return tokens, ids

    def make_input_sequences(self, json_path, sequence_length=1024):
        with open(json_path, "r") as f:
            ids = json.load(f)

        # split list of input tokens into list of elements of size max_len
        if sequence_length:
            logging.info(
                "Chunking samples in self.x to length of {}".format(sequence_length)
            )
            self.x = list(
                self.chunk_list(
                    ids,
                    n=sequence_length,
                    bos=self.tokenizer.bos_token_id,
                    eos=self.tokenizer.eos_token_id,
                )
            )
        elif sequence_length is None:
            logging.info("Not doing any chunking to .x")
            self.x = ids

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.tensor(self.x[index])


# wrapper function called when script is called
def runtime_code():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_tokens", type=str, help="path to training dataset")
    parser.add_argument("--valid_tokens", type=str, help="path to validation dataset")
    parser.add_argument("--test_tokens", type=str, help="path to test dataset")
    parser.add_argument(
        "--train_set_sizes", type=str, help="size of the training set (in M tokens)"
    )
    parser.add_argument(
        "--train_tokenizer",
        action="store_true",
        default=False,
        help="boolean, whether or not to train BPE tokenizer"
        "on wikitext-103 (default: false)",
    )
    parser.add_argument("--tokenizer_train_tokens", type=str)
    parser.add_argument(
        "--tokenizer_savedir",
        type=str,
        help="folder where merges.txt and vocab.txt are saved",
    )
    parser.add_argument("--train_savename", type=str)
    parser.add_argument("--valid_savename", type=str)
    parser.add_argument("--test_savename", type=str)
    parser.add_argument(
        "--savedir", type=str, help="path to where the tokenized dataset is saved"
    )

    args = parser.parse_args()

    # load the pretrained tokenizer or traine one from scracth
    # max len is not crucial here as we only want to retokenize the dataset, not
    # create the final sequences
    if os.path.exists(args.tokenizer_savedir) and os.path.isdir(args.tokenizer_savedir):
        if not os.listdir(args.tokenizer_savedir) and args.train_tokenizer:
            logging.info("Training BPE tokenizer...")
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(
                files=[
                    args.tokenizer_train_tokens,
                    args.valid_tokens,
                    args.test_tokens,
                ],
                vocab_size=28439,
                min_frequency=2,
                special_tokens=["<|endoftext|>", "<pad>"],
            )

            # Save files to disk
            logging.info(
                "Saving merges.txt and vocab.json to {}".format(args.tokenizer_savedir)
            )
            tokenizer.save_model(args.tokenizer_savedir)

        elif not os.listdir(args.tokenizer_savedir) and not args.train_tokenizer:
            raise ValueError("Tokenizer directory is empty. Use --train_tokenizer.")

        else:
            logging.info("Loading tokenizer from {}".format(args.tokenizer_savedir))
            tokenizer = GPT2TokenizerFast.from_pretrained(
                args.tokenizer_savedir, max_len=1024
            )
    else:
        raise Exception("{} directory doesn't exist".format(args.tokenizer_savedir))

    # now retokenize wikitext and save
    suffix = ".bpe.json"
    if args.train_tokens:
        train_ds = WikiTextDataset(tokenizer=tokenizer)
        train_ds.retokenize_txt(
            path=args.train_tokens,
            save_retokenized=os.path.join(args.savedir, args.train_savename + suffix),
        )

    # validation set
    if args.valid_tokens:
        eval_ds = WikiTextDataset(tokenizer=tokenizer)
        eval_ds.retokenize_txt(
            path=args.valid_tokens,
            save_retokenized=os.path.join(args.savedir, args.valid_savename + suffix),
        )

    # validation set
    if args.test_tokens:
        test_ds = WikiTextDataset(tokenizer=tokenizer)
        test_ds.retokenize_txt(
            path=args.test_tokens,
            save_retokenized=os.path.join(args.savedir, args.test_savename + suffix),
        )


if __name__ == "__main__":
    runtime_code()
