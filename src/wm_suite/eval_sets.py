LINZEN2016 = "/scratch/ka2773/project/lm-mem/sv_agr/linzen2016/linzen2016_english.json"
LAKRETZ2021_SHORT = (
    "/scratch/ka2773/project/lm-mem/sv_agr/lakretz2021/short_nested_outer_english.json"
)
LAKRETZ2021_LONG = (
    "/scratch/ka2773/project/lm-mem/sv_agr/lakretz2021/long_nested_outer_english.json"
)

sva_linzen = {"linzen2016": {"path": LINZEN2016, "comment": None}}
sva_lakretz = {
    "lakretz2021_ss": {"path": LAKRETZ2021_SHORT, "comment": "singular_singular"},
    "lakretz2021_sp": {"path": LAKRETZ2021_SHORT, "comment": "singular_plural"},
    "lakretz2021_ps": {"path": LAKRETZ2021_SHORT, "comment": "plural_singular"},
    "lakretz2021_pp": {"path": LAKRETZ2021_SHORT, "comment": "plural_plural"},
    "lakretz2021_sss": {
        "path": LAKRETZ2021_LONG,
        "comment": "singular_singular_singular",
    },
    "lakretz2021_ssp": {
        "path": LAKRETZ2021_LONG,
        "comment": "singular_singular_plural",
    },
    "lakretz2021_ppp": {"path": LAKRETZ2021_LONG, "comment": "plural_plural_plural"},
    "lakretz2021_pps": {"path": LAKRETZ2021_LONG, "comment": "plural_plural_singular"},
    "lakretz2021_sps": {
        "path": LAKRETZ2021_LONG,
        "comment": "singular_plural_singular",
    },
    "lakretz2021_psp": {"path": LAKRETZ2021_LONG, "comment": "plural_singular_plural"},
    "lakretz2021_spp": {"path": LAKRETZ2021_LONG, "comment": "singular_plural_plural"},
    "lakretz2021_pss": {
        "path": LAKRETZ2021_LONG,
        "comment": "plural_singular_singular",
    },
}


sva_short_labels = {
    "singular_singular": "SS",
    "singular_plural": "SP",
    "plural_singular": "PS",
    "plural_plural": "PP",
    "singular_singular_singular": "SSS",
    "plural_plural_plural": "PPP",
    "singular_singular_plural": "SSP",
    "plural_plural_singular": "PPS",
    "singular_plural_singular": "SPS",
    "plural_singular_plural": "PSP",
    "singular_plural_plural": "SPP",
    "plural_singular_singular": "PSS",
}


# a helper variable to filter if a dependency is short or long
sva_long_deps = [
    "singular_singular_singular",
    "singular_singular_plural",
    "plural_plural_plural",
    "plural_plural_singular",
    "singular_plural_singular",
    "plural_singular_plural",
    "singular_plural_plural",
    "plural_singular_singular",
]


sva_heterogenous = [
    "plural_singular",
    "singular_plural",
    "singular_singular_plural",
    "plural_plural_singular",
    "singular_plural_singular",
    "plural_singular_plural",
    "singular_plural_plural",
    "plural_singular_singular",
]
