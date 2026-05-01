import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import os

num_epochs = 10

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 512) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)          
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x))) 
            x = torch.flatten(x, 1) 
            
            x = F.relu(self.fc1(x))
            x = self.dropout(x) 
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
if __name__ == "__main__":
    train_dir="train/train"
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((28, 28)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])
    trainset = datasets.ImageFolder(train_dir, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=32,
                                            shuffle=True)

    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    num_checks = len([f for f in os.listdir("CNN model/checkpoints/")])
    checkpoint_path = f"CNN model/checkpoints/cnn_rules_{num_checks}.pt"
    loaded_checkpoint = False
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        loaded_checkpoint = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    if not loaded_checkpoint:
        for epoch in range(num_epochs):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        checkpoint = {"model_state_dict": net.state_dict(), "class_to_idx": trainset.class_to_idx}
        torch.save(checkpoint, checkpoint_path)
    val_dir="val/val"
    val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    valset = datasets.ImageFolder(val_dir, 
                                transform=val_transform)

    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=32,
                                            shuffle=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on validation images: {100 * correct // total} %')