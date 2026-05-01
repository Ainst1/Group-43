import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import cv2
import numpy as np
import os

num_epochs = 10

def binary_thresholding(img):
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    _, thresh_binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
    
    return thresh_binary
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5) 
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 60 * 60, 512) 
        self.fc2 = nn.Linear(512, 256)          
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) # Using the new layer
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == "__main__":
    train_dir="train/train"
    transform = transforms.Compose(
        [transforms.Resize(255),
        transforms.Lambda(binary_thresholding),
        transforms.ToTensor()])

    trainset = datasets.ImageFolder(train_dir, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=32,
                                            shuffle=True)

    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    checkpoint_path = "cnn_rules.pt"
    loaded_checkpoint = False
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        loaded_checkpoint = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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

    valset = datasets.ImageFolder(val_dir, 
                                transform=transform)

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