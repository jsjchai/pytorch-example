import torch
from torch import autograd

a = torch.empty(2, 3)
print(a)

x = torch.rand(3, 5)
y = torch.rand(3, 5)
print(x + y)

b = torch.zeros(2, 2, dtype=torch.long)
print(b)

c = torch.tensor([1.0, 2.2])
print(c)

x = torch.tensor(1.0)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a ** 2 * x + b * x + c

print(a.grad, b.grad, c.grad)
grad = autograd.grad(y, [a, b, c])
print(grad[0], grad[1], grad[2])
