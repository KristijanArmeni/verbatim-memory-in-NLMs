[![CircleCI](https://circleci.com/gh/KristijanArmeni/verbatim-memory-in-NLMs.svg?style=shield)](https://app.circleci.com/pipelines/github/KristijanArmeni/verbatim-memory-in-NLMs)

# Characterizing Short-term Memory In Neural Language Models

![thumbnail](thumbnail_2x.png)

ArXiV preprint:  https://arxiv.org/abs/2210.13569

# Folder contents

- `./`         |  main scripts
- `/data`      |  scripts and .txt files for creating inputs
- `/models` |  code with subfolders that contain pytorch model classes and training scripts for awd_lstm and transformer.
- `/wm_suite` | code with script for evaluating and analysing models on the working memory test suite
- `/viz` |  .py script containing plotting subroutines (.ipynb notebook is not versioned)

Each subfolder contains a README.md which explains the contents.

# Citation

```
@misc{https://doi.org/10.48550/arxiv.2210.13569,
      doi = {10.48550/ARXIV.2210.13569},
      url = {https://arxiv.org/abs/2210.13569},
      author = {Armeni, Kristijan and Honey, Christopher and Linzen, Tal},
      keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},  
      title = {Characterizing Verbatim Short-Term Memory in Neural Language Models},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution 4.0 International}
}

```

# Licence

This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.