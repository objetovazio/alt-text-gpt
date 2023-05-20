
import tensorflow as tf
import tensorflow_hub as hub

import csv

def print_csv_file(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            print(', '.join(row))

ex_sentence = [
    "A young boy is playing with a hockey stick",
    "A child in a red jacket playing street hockey guarding a goal",
    "A young kid playing the goalie in a hockey rink",
    "A young male kneeling in front of a hockey goal with a hockey stick in his right hand. Hockey goalie boy in red jacket crouches by goal a with stick.",
    "Hockey goalie boy in red jacket crouches by goal a with stick",
    "A boy with a stick kneeling in front of a goalie net",
]

embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")
embeddings = embed(ex_sentence)
print(embeddings.shape)

for i in range(len(ex_sentence)):
    if i == 0: continue

    print("{} \n{}:".format(ex_sentence[0], ex_sentence[i]))
    print(tf.keras.losses.cosine_similarity(
        embeddings[0],
        embeddings[i],
        axis=-1
    ))
    print('---------')
