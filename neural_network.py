import torch.nn as nn 
import torch

class NeuralNetwork(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
          torch.nn.Linear(num_inputs, 30),
          torch.nn.ReLU(),
          torch.nn.Linear(30, 20),
          torch.nn.ReLU(),
          torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
      logits = self.layers(x)
      return logits
torch.manual_seed(123)
X = torch.rand((1, 50))
model = NeuralNetwork(50,3)
out=model(X)
print("Output:", out)
# print(model)
# num_params = sum(p.numel() for p in model.parameters())
# print("Number of parameters:",num_params)
# print(model.layers[0].weight)
with torch.no_grad():
    out=model(X)
    print("Output with no_grad:", out)
with torch.no_grad():
    out=torch.softmax(model(X), dim=1)
    print("Output with no_grad again softmax:", out)
