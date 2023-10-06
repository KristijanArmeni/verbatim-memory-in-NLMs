#%%
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wm_suite.viz.ablation.inputs import get_filenames

#%%
def load_data(datadir, which):

    filename = get_filenames(os.path.basename(__file__))[which]

    return pd.read_csv(os.path.join(datadir, filename), sep="\t")


#%%

def make_lineplots(datadir, which):

    data = load_data(datadir=datadir,
                 which=which)

    # infer step, this sould be fixed in wm_subj_agr
    data["step"] = data["tag"].str.split("_").str.len()
    data = data.sort_values(by=["step"])

    # add a new label
    plural_targets = ["PS", "PPS", "PSP", "PSS"]
    data["target"] = data.agr.apply(lambda x: "Plural" if x in plural_targets else "Singular")

    # convert to percentages
    data["acc"] = data["acc"] * 100


    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    linestyles_singular = ["-o", "-^", "-s", "-d"]
    linestyles_plural = ["--o", "--^", "--s", "--d"]

    tmpdat = data.loc[data.target == "Singular"].copy()
    for i, dep in enumerate(tmpdat.agr.unique()):
        y = tmpdat.loc[tmpdat.agr == dep].acc.to_numpy()
        x = tmpdat.loc[tmpdat.agr == dep].step.to_numpy()
        ax[0].plot(x, y, linestyles_singular[i], label=dep)
        ax[0].set_title("Correct verb is singular\n(e.g. The key that the men hold [is])")

    tmpdat = data.loc[data.target == "Plural"].copy()
    for i, dep in enumerate(tmpdat.agr.unique()):
        y = tmpdat.loc[tmpdat.agr == dep].acc.to_numpy()
        x = tmpdat.loc[tmpdat.agr == dep].step.to_numpy()
        ax[1].plot(x, y, linestyles_plural[i], label=dep)
        ax[1].set_title("Correct verb is plural\n(e.g. The keys that the man holds [are])")

    for a in ax:
        a.hlines(50, 1, len(x), linestyle="--", color="tab:grey", label="Chance level")

    ax[0].legend(title="Dependency type")
    ax[1].legend(title="Dependency type")

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="both", linestyle="--", alpha=0.5)
        a.set_ylim(0, a.get_ylim()[1])

    ax[0].set_ylabel("Accuracy (%)")
    fig.supxlabel("Number of ablated heads")
    fig.suptitle("Subject-verb agreement performance after ablating most effective heads from unconstrained search")

    plt.tight_layout()

    return fig

# %%

def make_barplot(datadir):

    data1 = load_data(datadir, which="unablated")
    data2 = load_data(datadir, which="rand-init")
    data3 = load_data(datadir, which="full-ablation")

    for d in (data1, data2, data3):
        d["acc"] = d["acc"] * 100
        plural_targets = ["PS", "PPS", "PSP", "PSS"]
        d["target"] = d.agr.apply(lambda x: "Plural" if x in plural_targets else "Singular")

    data_ = pd.concat([data1, data2, data3], axis=0)


    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    data_sing = {"SP": data_.loc[data_.agr == "SP"].acc.to_numpy(),
                "SSP": data_.loc[data_.agr == "SSP"].acc.to_numpy(),
                "SPS": data_.loc[data_.agr == "SPS"].acc.to_numpy(),
                "SPP": data_.loc[data_.agr == "SPP"].acc.to_numpy()}

    data_plur = {"PS": data_.loc[data_.agr == "PS"].acc.to_numpy(),
                "PPS": data_.loc[data_.agr == "PPS"].acc.to_numpy(),
                "PSP": data_.loc[data_.agr == "PSP"].acc.to_numpy(),
                "PSS": data_.loc[data_.agr == "PSS"].acc.to_numpy()}

    width = 0.15
    multipler = -1.5

    for label, values in data_sing.items():
        x = np.arange(len(values))
        offset = width * multipler
        ax[0].bar(x+offset, values, width, label=label)
        multipler += 1


    width = 0.15
    multipler = -1.5

    for label, values in data_plur.items():
        x = np.arange(len(values))
        offset = width * multipler
        ax[1].bar(x+offset, values, width, label=label)
        multipler += 1

    ax[0].set_title("Correct verb is singular\n(e.g. The key that the men hold [is])")
    ax[1].set_title("Correct verb is plural\n(e.g. The keys that the man holds [are])")

    ax[0].set_ylabel("Accuracy (%)")

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_xticks(np.arange(len(values)))
        a.set_xticklabels(["Unablated", "Random init.", "Full ablation"])
        a.legend(title="Dependency type")
        a.grid(axis="y", linestyle="--", alpha=0.5)

    fig.supxlabel("Model type")


    plt.tight_layout()

    return fig

# %%

def make_plot(datadir, which):

    if which == "targeted_ablation":

        fig = make_lineplots(datadir=datadir, which="targeted")

    elif which == "random_ablation":

        fig = make_lineplots(datadir=datadir, which="random")

    elif which == "model_types":

        fig = make_barplot(datadir=datadir)

    return fig

#%%
def main(input_args=None):

    import argparse

    parser = argparse.ArgumentParser(description='Plot results of the subjec-verb agreement experiment')
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--which', type=str, choices=["all", "targeted_ablation", "random_ablation", "model_types"])
    parser.add_argument('--savedir', type=str)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    dofig1, dofig2, dofig3 = False, False, False
    if args.which == "all":
        dofig1 = True
        dofig2 = True
        dofig3 = True
    elif args.which == "targeted_ablation":
        dofig1 = True
    elif args.which == "random_ablation":
        dofig2 = True
    elif args.which == "model_types":
        dofig3 = True
    
    if dofig1:
        fig = make_plot(datadir=args.datadir, which="targeted_ablation")
        plt.show()
    if dofig2:
        fig = make_plot(datadir=args.datadir, which="random_ablation")
        plt.show()
    if dofig3:
        fig = make_plot(datadir=args.datadir, which="model_types")
        plt.show()

    pass

#%%
if __name__ == '__main__':

    main()

