import os
import requests
import tiktoken
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
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
