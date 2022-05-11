import torch
import builtins
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary

from torch_vision.common.pytorchcv import *

data_train, data_test = load_mnist()

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),  # 784 inputs, 10 outputs
    nn.LogSoftmax(dim=1))

print('Digit to be predicted: ', data_train[0][1])
torch.exp(net(data_train[0][0]))

train_loader = torch.utils.data.DataLoader(data_train, batch_size=64)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64)

print("train_epoch", train_epoch(net, train_loader))
print("validate", validate(net, test_loader))
hist = train(net, train_loader, test_loader, epochs=5)

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(hist['train_acc'], label='Training acc')
plt.plot(hist['val_acc'], label='Validation acc')
plt.legend()
plt.subplot(122)
plt.plot(hist['train_loss'], label='Training loss')
plt.plot(hist['val_loss'], label='Validation loss')
plt.legend()
#plt.show()

weight_tensor = next(net.parameters())
fig, ax = plt.subplots(1, 10, figsize=(15, 4))
for i, x in enumerate(weight_tensor):
    ax[i].imshow(x.view(28, 28).detach())
