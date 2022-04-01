from collections import Counter, OrderedDict

import torch
import torchtext
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import DataLoader
import numpy as np
from torchtext.vocab import vocab

from text2vec import to_bow


# this collate function gets list of batch_size tuples, and needs to
# return a pair of label-feature tensors for the whole minibatch
def bowify(b):
    return (
        torch.LongTensor([t[0] - 1 for t in b]),
        torch.stack([to_bow(t[1], len(b)) for t in b])
    )


def train_epoch(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.NLLLoss(), epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    total_loss, acc, count, i = 0, 0, 0, 0
    for labels, features in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = loss_fn(out, labels)  # cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
        count += len(labels)
        i += 1
        if i % report_freq == 0:
            print(f"{count}: acc={acc.item() / count}")
        if epoch_size and count > epoch_size:
            break
    return total_loss.item() / count, acc.item() / count


train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')

train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=bowify, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=bowify, shuffle=True)

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
counter = Counter()
for (label, line) in train_dataset:
    counter.update(tokenizer(line))

sorted_by_freq_tuples = sorted(counter.items(), key=lambda y: y[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
VOC = vocab(ordered_dict)
vocab_size = len(VOC)

net = torch.nn.Sequential(torch.nn.Linear(16, 4), torch.nn.LogSoftmax(dim=1))
train_epoch(net, train_loader, epoch_size=15000)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
corpus = [
    'I like hot dogs.',
    'The dog ran fast.',
    'Its hot outside.',
]
bigram_vectorizer.fit_transform(corpus)
print("Vocabulary:\n", bigram_vectorizer.vocabulary_)
bigram_vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

counter = Counter()
for (label, line) in train_dataset:
    l = tokenizer(line)
    counter.update(torchtext.data.utils.ngrams_iterator(l, ngrams=2))

bi_vocab = torchtext.vocab.vocab(counter, min_freq=1)

print("Bigram vocabulary length = ", len(bi_vocab))

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vectorizer.fit_transform(corpus)
vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()
