# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:43:12 2023

@author: he_98
"""

# Object detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform object detection using brute force
def detect_traffic_signs(image):
    # Parameters for sliding window
    window_size = (80, 80)  # Size of the sliding window
    stride = 50  # Stride for window movement

    detected_signs = []

    # Get image dimensions
    height, width, _ = image.shape
    #print(height)
    #print(width)
    # Slide the window across the image
    for y in range(0, height - window_size[1], stride):
        #print(y)
        for x in range(0, width - window_size[0], stride):
            # Extract the current window from the image
            window = image[y:y+window_size[1], x:x+window_size[0]]
            #print(y)
            #print(x)
            #print(window)
            #print(window.shape)
            # Perform traffic sign classification on the window using your CNN model
            #predicted_class = 3
            predicted_class, confidence = classify_traffic_sign(window)

            # If the predicted class indicates a traffic sign, add it to the list of detected signs
            if predicted_class is not None:
                detected_signs.append((x, y, window_size[0], window_size[1], predicted_class, confidence))
    
    return detected_signs


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
# Load your trained CNN model
model = CNNModel(num_classes)  # Create an instance of your CNN model
model.load_state_dict(torch.load('model1.pth'))  # Load the trained model state dict
model.eval()  # Set the model to evaluation mode

# Function to classify traffic signs using your pre-existing CNN model
def classify_traffic_sign(image):
    # Preprocess the image (resize, normalize, etc.) before passing it to the model
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = Image.fromarray(image)  # Convert NumPy array to PIL image
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = predicted.item()
    
    confidence_threshold = 0.98  # Adjust the threshold as needed

    if confidence.item() > confidence_threshold:
        print(f"Predicted label: {predicted_label}, Prob: {confidence.item()}")
        return predicted_label, confidence.item()
    else:
        print("No traffic sign detected.")
        return None, None


# Load and process the input image
input_image = cv2.imread('stop3.jpg')
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Perform object detection using brute force
detected_boxes = detect_traffic_signs(input_image_rgb)

# Classify the detected traffic signs using your CNN model
for box in detected_boxes:
    x, y, w, h, predicted_class, confidence = box

    # Extract the region of interest (ROI) from the original image
    roi = input_image_rgb[y:y+h, x:x+w]

    # Print the predicted class label
    print("Detected Traffic Sign:", predicted_class, "Prob:", confidence)

    # Draw the bounding box and display the image
    cv2.rectangle(input_image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(input_image_rgb)
plt.axis('off')
plt.show()
