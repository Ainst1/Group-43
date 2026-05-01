import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import csv
from pathlib import Path
from main import Net

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

TEST_DIR = "test/test"
CHECKPOINT_PATH = "CNN model/checkpoints/cnn_rules_5_best.pt"
num_csv = len([f for f in os.listdir("CNN model/predictions/")])
OUTPUT_CSV = f"CNN model/predictions/my_submission_{num_csv}.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net().to(device)
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval() 
    print(f"Loaded model from {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"Could not find {CHECKPOINT_PATH}")

results = []
image_files = [f for f in os.listdir(TEST_DIR)]

with torch.no_grad():
    for filename in image_files:
        image_id = Path(filename).stem 
        
        img_path = os.path.join(TEST_DIR, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) 
        
        outputs = net(img_tensor)
        _, predicted = torch.max(outputs, 1)
        category = predicted.item()
        
        results.append([image_id, category])

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Id', 'Category'])
    writer.writerows(results)

print(f"Predictions saved to {OUTPUT_CSV}")