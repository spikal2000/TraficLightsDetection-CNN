# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:58:44 2023

@author: he_98
"""

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out

num_classes = 43
model = CNNModel(num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

image_path = 'stop3Edit.jpg'

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)  # Add a batch dimension

with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)


confidence_threshold = 0.85  # Adjust the threshold as needed

if confidence.item() > confidence_threshold:
    predicted_label = predicted.item()
    print(f"Predicted label: {predicted_label}, Probability: {confidence.item()}")
else:
    print("No traffic sign detected.")