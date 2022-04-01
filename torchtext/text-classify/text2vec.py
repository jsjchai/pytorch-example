import torch
import torchtext
import os
import collections


def encode(x):
    return [vocab.stoi[s] for s in tokenizer(x)]


os.makedirs('./data', exist_ok=True)

train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
classes = ['World', 'Sports', 'Business', 'Sci/Tech']
next(iter(train_dataset))

for i, x in zip(range(5), train_dataset):
    print(f"**{classes[x[0]]}** -> {x[1]}")

train_dataset = list(train_dataset)
test_dataset = list(test_dataset)

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
tokenizer('He said: hello')

counter = collections.Counter()
for (label, line) in train_dataset:
    counter.update(tokenizer(line))
vocab = torchtext.vocab.Vocab(counter, min_freq=1)

vocab_size = len(vocab)
print(f"Vocab size if {vocab_size}")

encode('I love to play with my words')
