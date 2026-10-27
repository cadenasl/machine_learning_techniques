import torch.nn.functional as F
import torch
from torch.optim.sgd import SGD
from neural_network import NeuralNetwork
import compute_accuracy
from toyDataset import train_loader, test_loader, x_train,y_train


torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
    
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)
        
        loss = F.cross_entropy(logits, labels) # Loss function
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    # Optional model evaluation
    model.eval()

with torch.no_grad():
    outputs = model(x_train)

print("Outputs no gradient:", outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print("Outputs probabilities:", probas)

predictions = torch.argmax(probas, dim=1)
print("Predictions from logits:", predictions)
print(predictions)
print("Predictions match y_train:", predictions == y_train)
print("Number of correct predictions:", torch.sum(predictions == y_train))
print("Accuracy on training set:", compute_accuracy.compute_accuracy(model, train_loader))
print("Accuracy on test set:", compute_accuracy.compute_accuracy(model, test_loader))
# torch.save(model.state_dict(), "model.pth")
# model = NeuralNetwork(2, 2) # needs to match the original model exactly
print(model.load_state_dict(torch.load("model.pth", weights_only=True)))