import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import os

NUM_EPOCHS = 20
LOAD_CHECKPOINT = False
LOAD_CHECKPOINT_PATH = "CNN model/checkpoints/cnn_rules_5_best.pt"
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
        self.fc1 = nn.Linear(64 * 4 * 4, 128) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)          
        for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x))) 
            x = torch.flatten(x, 1) 
            
            x = F.relu(self.fc1(x))
            x = self.dropout(x) 
            x = F.relu(self.fc2(x))
            return x
if __name__ == "__main__":
    train_dir="train/train"
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])
    trainset = datasets.ImageFolder(train_dir, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
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
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    num_checks = len([f for f in os.listdir("CNN model/checkpoints/")])
    checkpoint_path = f"CNN model/checkpoints/cnn_rules_{num_checks}.pt"
    best_checkpoint_path = f"CNN model/checkpoints/cnn_rules_{num_checks}_best.pt"
    #best_checkpoint_path = f"CNN model/checkpoints/cnn_rules_5_best.pt"
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {LOAD_CHECKPOINT_PATH.split("/")[-1]} checkpoint")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    steps_per_epoch = len(trainloader)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if LOAD_CHECKPOINT and os.path.isfile(f"CNN model/checkpoint_scores/{best_checkpoint_path.split("/")[-1].split(".")[0]}.txt"):
         print("found best val loss score file")
         with open(f"CNN model/checkpoint_scores/{best_checkpoint_path.split("/")[-1].split(".")[0]}.txt","r") as file:
            best_val_loss = float(file.readlines()[0])
    else:
        if LOAD_CHECKPOINT:
            print("didn't find best val loss score file")
        best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valloader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"model_state_dict": net.state_dict()}, best_checkpoint_path)
            with open(f"CNN model/checkpoint_scores/{best_checkpoint_path.split("/")[-1].split(".")[0]}.txt","w") as file:
                 file.write(str(best_val_loss))
            print(f"Epoch {epoch+1}: New best model saved! Loss: {avg_val_loss:.4f}")


    print('Finished Training')

    #checkpoint = {"model_state_dict": net.state_dict(), "class_to_idx": trainset.class_to_idx}
    #torch.save(checkpoint, checkpoint_path)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on validation images: {(100 * correct / total):.7f} %')