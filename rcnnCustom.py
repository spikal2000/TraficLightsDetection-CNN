# Step 1: Import necessary libraries
import torch
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pandas as pd
import cv2
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Step 2: Load and preprocess data
train_df = pd.read_csv('C:/Users/spika/Desktop/ACG/MLA/Project/Traffic_Sign_Dataset/Traffic_Sign_Dataset/Train.csv')
test_df = pd.read_csv('C:/Users/spika/Desktop/ACG/MLA/Project/Traffic_Sign_Dataset/Traffic_Sign_Dataset/Test.csv')

train_df['Path'] = 'C:/Users/spika/Desktop/ACG/MLA/Project/Traffic_Sign_Dataset/Traffic_Sign_Dataset/' + train_df['Path']
test_df['Path'] = 'C:/Users/spika/Desktop/ACG/MLA/Project/Traffic_Sign_Dataset/Traffic_Sign_Dataset/' + test_df['Path']

# Step 3: Define a custom Dataset class
class TrafficSignsDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img = cv2.imread(row['Path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = Image.fromarray(img)

        # Resize image
        img = F.resize(img, [224, 224])

        # Convert back to numpy and normalize
        img = np.array(img) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        boxes = torch.tensor([[row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']]], dtype=torch.float32)

        labels = torch.tensor([row['ClassId']], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target


# Collation function to be used with the dataloader
def collate_fn(batch):
    return tuple(zip(*batch))

# Initialize the datasets and dataloaders
train_data = TrafficSignsDataset(train_df)
test_data = TrafficSignsDataset(test_df)

# Initialize the datasets and dataloaders with increased batch size
batch_size = 8

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

# Step 4: Initialize the model and specify the loss function, optimizer, and learning rate scheduler
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(train_df['ClassId'].unique()))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define loss function, optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Step 5: Train the model
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")
    model.train()

    # Create a progress bar
    loop = tqdm(train_dataloader, leave=True)

    for images, targets in loop:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update progress bar
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=losses.item())

    # update the learning rate
    lr_scheduler.step()

# Step 6: Evaluate the model
model.eval()
for images, targets in test_dataloader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    with torch.no_grad():
        prediction = model(images)
        # print prediction to see the result
        print(prediction)

# Save the model
torch.save(model.state_dict(), 'model_weights.pth')
print("Model saved")
