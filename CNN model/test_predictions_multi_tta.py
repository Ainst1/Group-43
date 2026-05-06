import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import csv
from pathlib import Path
from main import Net 

CHECKPOINT_PATHS = [
    "CNN model/checkpoints/cnn_rules_17_best.pt",
    "CNN model/checkpoints/cnn_rules_18_best.pt",
]
TTA_STEPS = 10
TEST_DIR = "test/test"
num_csv = len([f for f in os.listdir("CNN model/predictions/")])
OUTPUT_CSV = f"CNN model/predictions/my_submission_{num_csv}.csv"
tta_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        #transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = []
for path in CHECKPOINT_PATHS:
    if os.path.exists(path):
        net = Net().to(device)
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval() 
        models.append(net)
        print(f"Successfully loaded model from {path}")
    else:
        print(f"WARNING: Could not find {path}. Skipping.")

if not models:
    raise RuntimeError("No models were successfully loaded. Check your paths.")
results = []
image_files = [f for f in os.listdir(TEST_DIR)]

with torch.no_grad():
    for filename in image_files:
        image_id = Path(filename).stem 
        
        img_path = os.path.join(TEST_DIR, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) 
        
        ensemble_probs = torch.zeros((1, 10)).to(device)
        
        for net in models:
            img_tensor = transform(img).unsqueeze(0).to(device)
            ensemble_probs += F.softmax(net(img_tensor), dim=1)

            for _ in range(TTA_STEPS - 1):
                img_tensor = tta_transform(img).unsqueeze(0).to(device)
                ensemble_probs += F.softmax(net(img_tensor), dim=1)
        ensemble_probs /= len(models) * TTA_STEPS
        _, predicted = torch.max(ensemble_probs, 1)
        category = predicted.item()
        
        results.append([image_id, category])

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Id', 'Category'])
    writer.writerows(results)

print(f"Ensemble predictions saved to {OUTPUT_CSV}")