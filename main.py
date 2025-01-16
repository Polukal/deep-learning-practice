import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import certifi

# Visualization tools
import torchvision
import torchvision.transforms as transforms
from utils import train, validate

# Set the SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

# Create data directory if it doesn't exist
os.makedirs("./data", exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load datasets with error handling
try:
    train_set = torchvision.datasets.MNIST("./data/", train=True, download=True, transform=transform)
    valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True, transform=transform)
except RuntimeError as e:
    print(f"First attempt failed: {e}")
    ssl._create_default_https_context = ssl._create_unverified_context
    train_set = torchvision.datasets.MNIST("./data/", train=True, download=True, transform=transform)
    valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True, transform=transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# Calculate dataset sizes for accuracy computation
train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)


# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Create model instance
    model = SimpleCNN()
    model = model.to(device)

    # Use PyTorch 2.0 compilation for better performance
    model = torch.compile(model)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        train(model, train_loader, train_N, device, optimizer, loss_function)
        validate(model, valid_loader, valid_N, device, loss_function)

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')