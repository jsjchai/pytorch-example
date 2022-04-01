import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 图像张量
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # 设置网络中的全连接层
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(3, 28, 28)
print(input_image.size())

# 将每个2D 28*28图片转化成784像素值
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 使用存储的权重和偏差对输入应用线性变换
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# 非线性激活在模型的输入和输出之间创建复杂的映射。它们应用于线性变换后引入非线性，帮助神经网络学习各种各样的现象。在这个模型中，我们使用了神经网络。线性层之间的ReLU，但还有其他的激活来引入非线性
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential是一个有序的模块容器。数据按照定义的顺序通过所有模块。您可以使用顺序容器来组合一个快速网络
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

#
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# 模型参数
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
