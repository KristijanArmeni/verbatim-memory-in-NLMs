# -*- coding: utf-8 -*-
"""
script to load word lists and export to latex table via pandas

Created on Fri Apr  9 17:08:10 2021

@author: karmeni1
"""

import os
import json
import pandas as pd

root = os.path.join(os.environ["homepath"], "project", "lm-mem", "src", "data")
savedir = os.path.join(os.environ["homepath"], "project", "lm-mem", "tables")

# make sure pandas prints full strings
# for exporting in .tex
pd.set_option("max_colwidth", None)

semantic = "categorized"
arbitrary = "random"

map_names = {
    "categorized": "semantic",
    "random": "arbitrary",
    }

for l in ["random", "categorized"]:
    
    fname = os.path.join(root, "{}_lists.json".format(l))
    
    with open(fname, "r") as f:
        lists = json.load(f)
    
    strings = [", ".join(nouns) + "." for nouns in lists["n10"]]
    
    # convert to dataframe
    df = pd.DataFrame(strings, columns=["list"])
    
    # index rows with one and give it a name
    df.index += 1
    
    cap = "Lists of {} nouns used in present experiments.".format(map_names[l])
    tex = df.to_latex(caption=cap, label="tab:{}_nouns".format(map_names[l]))
    
    # now save as .tex file
    fname = os.path.join(savedir, "{}_nouns.tex".format(map_names[l]))
    print("Writing {}".format(fname))
    with open(fname, "w") as f:
        
        f.writelines(tex)