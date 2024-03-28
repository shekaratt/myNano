"""
Prepare the Wodehouse dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the PG Wodehouse text from the public domain Project Guttenberg dataset
# wodehouse.txt made up of cocatenation of 6 books
# The Man with two left feet  - https://www.gutenberg.org/cache/epub/7471/pg7471.txt
# Leave it to Psmith          - https://www.gutenberg.org/cache/epub/60067/pg60067.txt
# Psmith in the City          - https://www.gutenberg.org/cache/epub/6753/pg6753.txt
# Psmith, Journalist          - https://www.gutenberg.org/cache/epub/2607/pg2607.txt
# Mike                        - https://www.gutenberg.org/cache/epub/7423/pg7423.txt
# Mike and Psmith             - https://www.gutenberg.org/cache/epub/10586/pg10586.txt
#
input_file_path = os.path.join(os.path.dirname(__file__), 'PGWodehouse.txt')
if not os.path.exists(input_file_path):
    books = ["https://www.gutenberg.org/cache/epub/7471/pg7471.txt",
             "https://www.gutenberg.org/cache/epub/60067/pg60067.txt",
             "https://www.gutenberg.org/cache/epub/6753/pg6753.txt",
             "https://www.gutenberg.org/cache/epub/2607/pg2607.txt",
             "https://www.gutenberg.org/cache/epub/7423/pg7423.txt",
             "https://www.gutenberg.org/cache/epub/10586/pg10586.txt"]
    for book_url in books :
        with open(input_file_path, 'a', encoding='utf-8') as f:
            f.write(requests.get(book_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
