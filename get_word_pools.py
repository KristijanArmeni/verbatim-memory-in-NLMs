from requests import get
import os

# wrapper for getting urls
def download_to(url, file_name):

    with open(file_name, "wb") as file:

        response = get(url)
        file.write(response.content)


# we use 2 noun pools for now
noun_pools ={
    "nouns_categorized": "http://memory.psych.upenn.edu/files/wordpools/catwpool.txt",
    "toronto_freq": "http://memory.psych.upenn.edu/files/wordpools/nouns.txt"
}

# get the files
filepath = "C:\\Users\\Kristijan\\Documents\\projects\\jhu1"
for pool in noun_pools.keys():

    destination = os.path.join(filepath, "data", pool + ".txt")
    print("Fetching {}...".format(noun_pools[pool]))
    print("Saving to {}...".format(destination))

    download_to(url=noun_pools[pool],
                file_name=destination)