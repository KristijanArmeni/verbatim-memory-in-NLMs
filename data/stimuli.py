import random

# define prefixes
prefixes = {
    "sce1": "Before the meeting, Mary brainstormed the following list of words:"
}

# define prompts
prompts_word_prediction = {
    "sce1-1": "After the meeting, Mary read the list again:",

    "sce1-3": " ".join(["After the meeting, Mary went for a walk. It was a busy day and she had to relax.",
                        "It was a warm sunny day and the flowers in the park were blooming.",
                        "When she got back, she read the list again:"]),

    "sce1-5": " ".join(["After the meeting, Mary went for a walk. It was a busy day and she had to relax.",
                        "It was a warm sunny day and the flowers in the park were blooming.",
                        "While she was walking, she listened to the wonderful bird songs.",
                        "During the walk, Mary could not stop thinking about the meeting.",
                        "She was thinking about the discussions she had with her coworkers.",
                        "Luckily, she met her neighbours Joanna and Ryan. They talked a little bit and then ",
                        "Joanna and Ryan invited her to come over for a cup of coffee.",
                        "Mary said she had time over the weekend. After they said goodbye, Mary started walking",
                        "towards home. When she got back, she read the list again:"])
}

# define prefixes
prefixes_perplexity = {
    "sce1": "Before the meeting, Mary composed the following list of words:",
}
# Define a meainigful probe
a = "The landlady informed me that he had left the house shortly after eight o’clock" \
    " in the morning. I sat down beside the fire, however, with the intention of awaiting him."
random.seed(123)
long2 = random.sample(a.split(), len(a.split()))

prompts_perplexity ={
     "sce1-1": "After the meeting, Mary read the list again:",

     "sce1-2": "After the meeting, Mary went for a walk. It was a busy day and she had to relax." +
               "When she got back, she read the list again:",

     "sce1-3": "After the meeting, Mary went for a walk. It was a busy day and she had to relax." +
               "It was a warm sunny day and the flowers in the park were blooming." +
               "When she got back, she read the list again:",

     "sce1-4": "After the meeting, Mary went for a walk. It was a busy day and she had to relax." +
               "It was a warm sunny day and the flowers in the park were blooming." +
               "While she was walking, she listened to the wonderful bird songs." +
               "During the walk, Mary could not stop thinking about the meeting." +
               "When she got back, she read the list again:",

     "sce1-5": "After the meeting, Mary went for a walk. It was a busy day and she had to relax." +
               "It was a warm sunny day and the flowers in the park were blooming." +
               "While she was walking, she listened to the wonderful bird songs." +
               "During the walk, Mary could not stop thinking about the meeting." +
               "She was thinking about the discussions she had with her coworkers." +
               "Luckily, she met her neighbours Joanna and Ryan. They talked a little bit and then" +
               "Joanna and Ryan invited her to come over for a cup of coffee." +
               "Mary said she had time over the weekend. After they said goodbye, Mary started walking" +
               "towards home. When she got back, she read the list again:"
}