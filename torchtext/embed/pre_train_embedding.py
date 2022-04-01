import torch
import torchtext

from embeddingClassifier import *
import gensim.downloader as api

from torchnlp import load_dataset, train_epoch_emb, padify, encode


def offsetify(b):
    # first, compute data tensor from all sequences
    x = [torch.tensor(encode(t[1], voc=vocab)) for t in b]  # pass the instance of vocab to encode function!
    # now, compute the offsets by accumulating the tensor of sequence lengths
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)
    return (
        torch.LongTensor([t[0] - 1 for t in b]),  # labels
        torch.cat(x),  # text
        o
    )


w2v = api.load('word2vec-google-news-300')
embed_size = len(w2v.get_vector('hello'))
print(f'Embedding size: {embed_size}')

train_dataset, test_dataset, classes, vocab = load_dataset()
vocab_size = len(vocab)

net = EmbeddingClassifier(vocab_size, embed_size, len(classes))

print('Populating matrix, this will take some time...', end='')
found, not_found = 0, 0
for i, w in enumerate(vocab.itos):
    try:
        net.embedding.weight[i].data = torch.tensor(w2v.get_vector(w))
        found += 1
    except:
        net.embedding.weight[i].data = torch.normal(0.0, 1.0, (embed_size,))
        not_found += 1

print(f"Done, found {found} words, {not_found} words missing")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=padify, shuffle=True)
train_epoch_emb(net, train_loader, lr=4, epoch_size=25000)

vocab = torchtext.vocab.GloVe(name='6B', dim=50)

# get the vector corresponding to kind-man+woman
qvec = vocab.vectors[vocab.stoi['king']] - vocab.vectors[vocab.stoi['man']] + 1.3 * vocab.vectors[vocab.stoi['woman']]
# find the index of the closest embedding vector
d = torch.sum((vocab.vectors - qvec) ** 2, dim=1)
min_idx = torch.argmin(d)
# find the corresponding word
vocab.itos[min_idx]

net = EmbeddingClassifier(len(vocab), len(vocab.vectors[0]), len(classes))
net.embedding.weight.data = vocab.vectors
net = net.to(device)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=offsetify, shuffle=True)
train_epoch_emb(net, train_loader, lr=4, epoch_size=25000)
