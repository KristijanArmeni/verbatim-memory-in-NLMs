import numpy as np
from nltk import word_tokenize
import string
import os, json

# define prefixes
prefixes = {
    "sce1": {"1": "Before the meeting, Mary wrote down the following list of words:"},
    "sce1rnd": {"1": "Before the meeting, Mary wrote down the following list of words:"},
    "sce2": {"1": "Before the meeting, Mary wrote down the following list of words:"},
    "sce3": {"1": "Before the meeting, Mary wrote down the following lists of words. One was:"}
}

# define prefixes for the n-gram experiment
prefixes_repeated_ngrams = {
    "sce1": {"1": "In the newspaper, somebody wrote the following words:"},
}

# ===== VIGNETTES =====#

# ===== MARY SCENARIO =====#

a1 = ["After the meeting, she took a break and had a cup of coffee."]

a2 = ["After the meeting, Mary went for a walk.",
      "It was a busy day and she needed a break.",
      "Outside was really beautiful and warm and the flowers in the park were blooming."]

a3 = ["While she was walking, she listened to the wonderful bird songs.",
     "During the walk, Mary could not stop thinking about the meeting.",
     "She was thinking about the discussions she had with her coworkers.",
     "Luckily, she met her neighbors Sarah and Ryan and they talked briefly.",
     "After they said goodbye, Mary started walking towards home."]

a4 = ["The couple has just moved to the area from a different city.",
      "Mary thought they were very a lovely couple and made good company.",
      "They were just getting to know the neighborhood and this was their first time in the park.",
      "Mary was curious what were their first impressions of the town.",
      "The neighborhood felt very safe to them and they absolutely loved the park.",
      "This was only their second time visiting the park.",
      "There was so much to discover, so many winding paths and hidden gardens.",
      "After they said goodbye, Mary started walking towards home."]

a5 = ["It was not a big park by any means, but it offered a quiet refuge where one can escape the worries of everyday life.",
     "It also offered opportunities to do sports of all kinds.",
     "Young people from around the area played basketball, football, or volleyball.",
     "Others took part in outdoor workout sessions.",
     "Young families were going on a stroll with their children.",
     "Finally, there were so many people who brought their dogs for a walk.",
     "It was incredibly satisfying to see the joy our animal friends get when you throw them a ball.",
     "All this diversity of people and activities made a walk in this park a truly rewarding and relaxing daily routine.",
     "In fact, Sarah and Ryan were thinking of getting a dog.",
     "They have not fully decided yet but they really wanted to spend more time outdoors.",
     "Mary liked dogs as well, but she was more of a cat person herself.",
     "She and her husband had two cats.",
     "One was two and the other four years old.",
     "They were very independent and spent most of their time outdoors.",
     "Mary thought having an animal was a great idea.",
     "They talked for a little bit and then Sarah and Ryan invited her to come over for a cup of coffee.",
     "Mary said she had time over the weekend.",
     "After they said goodbye, Mary started walking towards home."]


# ===== INCONGRUENT OCEAN SCENARIO ==== #

b1 = ["There is a voice in the waters of the great sea. It calls to man continually."]

b2 = ["Sometimes it thunders in the tempest, when the waves leap high and strong and the wild winds shriek and roar.",
      "Sometimes it whispers in the calm, small voice, as if to solicit our regard."]

b3 = ["The sea has much to say; far more than could possibly be comprehended in one volume, however large.",
      "It tells us of the doings of man on its broad bosom, from the day in which he first ventured to paddle along shore "
      "to the day when he launched his great iron ship, "
      "and rushed out to sea."]

b4 = ["Before proceeding to the consideration of the wonders connected with and contained in the sea, "
      "we shall treat of the composition of the sea itself and of its extent, depth, and bottom.",
      "What is the sea made of?",
      "Salt water, is the ready reply that rises naturally to every lip.",
      "But to this we add the question, what is salt water?",
      "To these queries we give the following reply, which, we doubt not, will rather surprise some of our readers.",
      "The salt of the ocean varies considerably in different parts."]

b5 = ["Near the equator, the great heat carries up a larger proportion of water by evaporation than in the more temperate regions.",
     "Thus, as salt is not removed by evaporation, the ocean in the torrid zone is salter than in the temperate or frigid zones.",
     "The salts of the sea, and other substances contained in it, are conveyed there by the fresh water streams that pour into it from all the continent of the world",
     "Here, as these substances cannot be evaporated, they would accumulate to such a degree "
     "as to render the ocean uninhabitable by living creatures."
     "The operations of the ocean are manifold.",
     "But we cannot speak of these things without making passing reference to the operations of water, "
     "as that wonder-working agent of which the ocean constitutes but a part.",
     "Nothing in this world is ever lost or annihilated.",
     "As the ocean receives all the water that flows from the land, "
     "so it returns that water, fresh and pure, in the shape of vapour, to the skies.",
     "where, in the form of clouds, it is conveyed to those parts of the earth where its presence is most needed. "
     "After having gladdened the heart of man by driving his mills and causing his food to grow, "
     "it finds its way again into the sea: and thus the good work goes on with ceaseless regularity."]

# ===== SHORTER INTERVENING CONTEXT ===== #

c1 = ["And the other:"]

# ===== FUNCTIONS ===== #

def add_sections(chunks, skip_last_item=True):

    out = []

    if len(chunks) > 1:
        out += chunks[0]
        for chunk in chunks[1::]:
            if skip_last_item:
                out += chunk[0:-1]
            else:
                out += chunk[0::]
    else:
        out += chunks[0]
    return out


def permute_tokens(token_string):

    # set random seed
    rng = np.random.RandomState(12345)

    # lower case the first character prior to shuffling, such that it can't appear capitalized
    # in other positions, don't do that for proper names
    # also strip the final punctuation
    token_list = word_tokenize(token_string.strip("."))
    first = token_list[0]
    proper = ["Mary", "Sarah", "Ryan"]
    if first not in proper:
        token_list = [token_list[0].lower()] + token_list[1:]

    shuf = rng.permutation(token_list)

    if shuf[0] in string.punctuation:
        pnct = shuf[0]
        shuf[0] = shuf[1].capitalize()
        shuf[1] = pnct
    else:
        shuf[0] = shuf[0].capitalize()

    shuf = " ".join(shuf.tolist()).replace(" :", ":") \
        .replace(" ,", ",") \
        .replace(" .", ".")

    return shuf


def permute_sents(sent_list):
    # set random seed
    rng = np.random.RandomState(54321)

    out_sents = []
    for sent in sent_list:
        shuf = permute_tokens(sent)
        out_sents.append(shuf + ".")

    return rng.permutation(out_sents).tolist()


# define the prompts
sce1 = {"1": " ".join(a1),
        "2": " ".join(a2),
        "3": " ".join(add_sections((a2, a3), skip_last_item=True)),
        "4": " ".join(add_sections((a2, a3, a4), skip_last_item=True)),
        "5": " ".join(add_sections((a2, a3, a4, a5), skip_last_item=True))
    }

# create randomized version of sce1 by permuting prompts on a word level
l1 = permute_sents(a1)
l2 = permute_sents(a2)
l3 = permute_sents(add_sections((a2, a3), skip_last_item=True))
l4 = permute_sents(add_sections((a2, a3, a4), skip_last_item=True))
l5 = permute_sents(add_sections((a2, a3, a4, a5), skip_last_item=True))

sce1rnd = {
            "1": " ".join(l1),
            "2": " ".join(l2),
            "3": " ".join(l3),
            "4": " ".join(l4),
            "5": " ".join(l5),
}

# scenario
sce2 = {
    "1": " ".join(b1),
    "2": " ".join(b2),
    "3": " ".join(add_sections((a2, b3), skip_last_item=False)),
    "4": " ".join(add_sections((a2, b3, b4), skip_last_item=False)),
    "5": " ".join(add_sections((a2, b3, b4, b5), skip_last_item=False))
}

sce3 = {
        "1": c1[0],
        }

# Join the context and the prompt
prompt_string = "When she got back, she read the list again:"
for scen in (sce1, sce1rnd, sce2):
    for key in scen.keys():
        scen[key] = scen[key] + " " + prompt_string


prompts = {
    "sce1": sce1,
    "sce1rnd": sce1rnd,
    "sce2": sce2,
    "sce3": sce3,
}

if __name__ == "__main__":
    
    # save data for supplementary materials
    
    savedir = os.path.join(os.environ["homepath"],
                           "project",
                           "lm-mem",
                           "src",
                           "data")
    
    
    noun_list_template = ["$W_{1}, W_{2}, ..., W_{N}$\n"]
    
    format_text = lambda nouns, intervening_texts: \
                  ["\n".join(["$intervening\_text_{}$: ".format(i+1) + l + " " + 
                             nouns[0] for i, l in enumerate(intervening_texts)])]
    
    # intact scenario
    preface_string = prefixes["sce1"]["1"]
    intervening_texts = "\n\n".join([preface_string] + 
                                    noun_list_template +
                                    format_text(noun_list_template,
                                                list(prompts["sce1"].values())))
    
    fname = os.path.join(savedir, "intervening_text_intact.tex")
    print("Writing {}".format(fname))
    with open(fname, "w") as f:
        f.writelines(intervening_texts)
    
    
    # scrambled scneario
    preface_string = prefixes["sce1rnd"]["1"]
    intervening_texts = "\n\n".join([preface_string] + 
                                    noun_list_template +
                                    format_text(noun_list_template,
                                                list(prompts["sce1rnd"].values())))
    
    fname = os.path.join(savedir, "intervening_text_scrambled.tex")
    print("Writing {}".format(fname))
    with open(fname, "w") as f:
        f.writelines(intervening_texts)
    
    
    # incongruent scneario
    preface_string = prefixes["sce2"]["1"]
    intervening_texts = "\n\n".join([preface_string] + 
                                    noun_list_template + 
                                    format_text(noun_list_template,
                                                list(prompts["sce2"].values())))

    fname = os.path.join(savedir, "intervening_text_incongruent.tex")
    print("Writing {}".format(fname))
    with open(fname, "w") as f:
        f.writelines(intervening_texts)
        

    # short scenario
    preface_string = prefixes["sce3"]["1"]
    intervening_texts = "\n\n".join([preface_string] + 
                                    noun_list_template +
                                    format_text(noun_list_template,
                                                list(prompts["sce3"].values())))
    
    fname = os.path.join(savedir, "intervening_text_short.tex")
    print("Writing {}".format(fname))
    with open(fname, "w") as f:
        f.writelines(intervening_texts)
    
    
    # save as .json files as well
    fname_prefixes = os.path.join(savedir, "preface_text.json")
    fname_intervening_texts = os.path.join(savedir, "intervening_texts.json")
    
    print("Writing {}".format(fname_prefixes))
    with open(fname_prefixes, "w") as f:
        json.dump(prefixes, f)
    
    print("Writing {}".format(fname_intervening_texts))
    with open(fname_intervening_texts, "w") as f:
        json.dump(prompts, f)