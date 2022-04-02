import os
from collections import Counter, OrderedDict

import torch
import torchtext
from torchtext.vocab import vocab


def to_bow(text, bow_vocab_size):
    res = torch.zeros(bow_vocab_size, dtype=torch.float32)
    for i in encode(text):
        if i < bow_vocab_size:
            res[i] += 1
    return res


def encode(x):
    vec = [VOC[s] for s in tokenizer(x)]
    #print(vec)
    return vec


os.makedirs('./data', exist_ok=True)

train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='../data')
classes = ['World', 'Sports', 'Business', 'Sci/Tech']
next(iter(train_dataset))

for i, x in zip(range(5), train_dataset):
    print(f"**{classes[x[0]]}** -> {x[1]}")

train_dataset = list(train_dataset)
test_dataset = list(test_dataset)

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
tokenizer('He said: hello')

counter = Counter()
for (label, line) in train_dataset:
    counter.update(tokenizer(line))

sorted_by_freq_tuples = sorted(counter.items(), key=lambda y: y[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
VOC = vocab(ordered_dict)

vocab_size = len(VOC)
print(f"Vocab size if {vocab_size}")

encode('I love to play with my words')

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = [
    'I like hot dogs.',
    'The dog ran fast.',
    'Its hot outside.',
]
vectorizer.fit_transform(corpus)
vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

vocab_size = len(VOC)

print(to_bow(train_dataset[0][1], vocab_size))
