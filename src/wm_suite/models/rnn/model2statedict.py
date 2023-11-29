# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:11:45 2021

@authors: karmeni1, Gabriel Kressin
"""

import os
import torch
import argparse


def convert_model(model_file):
    model_object = torch.load(model_file, map_location="cpu")

    # check if filename is provided, else append weights to input file name for saving
    output_fname = model_file.replace(".pt", "_statedict.pt")

    # save the state dict only
    print("Saving {}".format(output_fname))
    torch.save(model_object.state_dict(), output_fname)


def convert_models(model):
    """Iterates (if possible) through dir and converts all models to statedicts"""
    if os.path.isdir(model):
        for model_file in os.listdir(model):
            if "state_dict" not in model_file:
                convert_model(os.path.join(model, model_file))
    else:
        convert_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="converts pytorch model files into their state_dict")
    parser.add_argument("model", type=str,
        help="path/to/models containing multiple lstm.pt files or single path/to/model.pt")

    args = parser.parse_args()

    convert_models(
        model=args.model
    )
