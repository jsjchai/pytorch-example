from torch_text.common.torchnlp import *
from embeddingClassifier import *
from embeddingBagClassifier import EmbeddingBagClassifier

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
