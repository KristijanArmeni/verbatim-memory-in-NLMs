# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:11:45 2021

@author: karmeni1
"""

import os
import model
import torch

source_dir = "C:\\Users\\karmeni1\\project\\lm-mem\\data\\checkpoints\\lstm"
checkpoints = ['LSTM_800_80m_a_15-d0.2.pt']

# do the work

for ckpname in checkpoints:

    in_fullfile = os.path.join(source_dir, ckpname)
    model_object = torch.load(in_fullfile, map_location="cpu")

    # check if filename is provided, else append weights to input file name for saving
    output_fname = ckpname.replace(".pt", "_statedict.pt")

    # save the state dict only
    out_fullfile = os.path.join(source_dir, output_fname)
    print("Saving {}".format(out_fullfile))
    torch.save(model_object.state_dict(), out_fullfile)
