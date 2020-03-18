from __future__ import print_function
import torch


#USING TENSORS

x = torch.empty(5,3)
print(x)

y = torch.rand(5,3)
print(y)

x = torch.zeros(5,3, dtype=torch.long)
print(x)

z = torch.tensor([5.5, 3])
print(z)

print(torch.add(y,y))


#AUTOGRAD
print()
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x+2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)


#GRADIENTS
out.backward()
print(x.grad)

