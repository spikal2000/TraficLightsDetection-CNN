# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:23:29 2023

@author: he_98
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


# Assuming you have downloaded the dataset and extracted it into a folder
train_data = pd.read_csv('Traffic_Sign_Dataset/Traffic_Sign_Dataset/Train.csv')
'''
training_data = []
for i in train_data['Path']:
    #print(i)
    #train_data['Path']
    path = 'Traffic_Sign_Dataset/Traffic_Sign_Dataset' + '/'+ i
    #print(path)
    training_data.append(path)
'''


#train_data = datasets.ImageFolder('Traffic_Sign_Dataset/Traffic_Sign_Dataset/Train.csv', transform=transforms.ToTensor())
#test_data = datasets.ImageFolder('path/to/test/folder', transform=transforms.ToTensor())

class TrafficSignsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.root_dir + '/' + self.data['Path'][idx]
        image = Image.open(image_path).convert('RGB')
        label = self.data['ClassId'][idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Set the path to your CSV file and image folder
csv_file_train = 'Traffic_Sign_Dataset/Traffic_Sign_Dataset/Train.csv'
csv_file_test = 'Traffic_Sign_Dataset/Traffic_Sign_Dataset/Test.csv'

image_folder = 'Traffic_Sign_Dataset/Traffic_Sign_Dataset'

# Define the transformation to apply to the training data
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Converts the images to PyTorch tensors
    # Add any additional transformations here
])

# Create the dataset
train_dataset = TrafficSignsDataset(csv_file=csv_file_train, root_dir=image_folder, transform=transform)
print(train_dataset)
test_dataset = TrafficSignsDataset(csv_file=csv_file_test, root_dir=image_folder, transform=transform)
print(test_dataset)

# Create a data loader to efficiently load the data during training
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define your CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 1 * 1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.classifier(x)
        return x


# Create an instance of your CNN model
num_classes = 43  # Replace with the number of classes in your dataset
model = CNNModel(num_classes)


# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your model
num_epochs = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

for epoch in range(num_epochs):
    # Training loop
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    validation_accuracy = correct / total
    # Print training/validation metrics for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Accuracy: {validation_accuracy}")

torch.save(model.state_dict(), 'model.pth')