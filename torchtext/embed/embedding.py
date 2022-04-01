import torch
import torchtext
import numpy as np
from torchnlp import *
from embeddingClassifier import *
from embeddingBagClassifier import EmbeddingBagClassifier


def padify(b):
    # b is the list of tuples of length batch_size
    #   - first element of a tuple = label,
    #   - second = feature (text sequence)
    # build vectorized sequence
    v = [encode(x[1]) for x in b]
    # first, compute max length of a sequence in this minibatch
    l = max(map(len, v))
    return (  # tuple of two tensors - labels and features
        torch.LongTensor([t[0] - 1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t), (0, l - len(t)), mode='constant', value=0) for t in v])
    )


def offsetify(b):
    # first, compute data tensor from all sequences
    x = [torch.tensor(encode(t[1])) for t in b]
    # now, compute the offsets by accumulating the tensor of sequence lengths
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)
    return (
        torch.LongTensor([t[0] - 1 for t in b]),  # labels
        torch.cat(x),  # text
        o
    )


def train_epoch_emb(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None,
                    report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss, acc, count, i = 0, 0, 0, 0
    for labels, text, off in dataloader:
        optimizer.zero_grad()
        labels, text, off = labels.to(device), text.to(device), off.to(device)
        out = net(text, off)
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


train_dataset, test_dataset, classes, vocab = load_dataset()
vocab_size = len(vocab)
print("Vocab size = ", vocab_size)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=padify, shuffle=True)

net = EmbeddingClassifier(vocab_size, 32, len(classes)).to(device)
print("train_loader1:")
train_epoch(net, train_loader, lr=1, epoch_size=25000)

train_loader_bag = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=offsetify, shuffle=True)
net = EmbeddingBagClassifier(vocab_size, 32, len(classes)).to(device)
print("train_loader2:")
train_epoch_emb(net, train_loader_bag, lr=4, epoch_size=25000)
