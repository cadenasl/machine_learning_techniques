import torch.nn.functional as F
import torch
y=torch.tensor([1.0,0.0])
x1=torch.tensor([1.0,1.0])
w1=torch.tensor([2.0,2.0])
b=torch.tensor([0.0,0.0])
z=x1*w1+b
a=torch.sigmoid(z)
loss=F.binary_cross_entropy(a, y)
