import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

if device == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def evaluate(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total

# Hyperparameters
EPOCHS = 10
LR = 0.05

model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x, y in train_loader:
        
        optimizer.zero_grad()        # clear gradients from previous batch before computing new ones

        
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    test_acc = evaluate(model, test_loader, device)
    train_losses.append(avg_loss)
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch+1:2d} | loss = {avg_loss:.4f} | test_acc = {test_acc:.4f}")

# Plot
epochs_range = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, train_losses, color='steelblue', linewidth=2, marker='o')
ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(epochs_range, test_accuracies, color='seagreen', linewidth=2, marker='o')
ax2.set_title('Test Accuracy', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True)

plt.suptitle('Learning Curves - MLP PyTorch on MNIST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()