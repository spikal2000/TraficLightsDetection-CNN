import cv2
import numpy as np
from ultralytics import YOLO

# def preprocess_for_your_model(image):
#     # Add your preprocessing steps here. For example, resizing the image to the
#     # input size of your model, normalization, etc.
#     pass

# Initialize the YOLO model
model = YOLO("yolov8m_custom.pt")

# Load your CNN model here
# your_model = load_your_model()

# Predict with YOLO
results = model.predict(source="0", show=True)

# Loop over the detections
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = [int(i) for i in box[:4]]  # Get the box coordinates
        # Extract the object patch
        obj_img = result.orig_img[y1:y2, x1:x2]
        # Preprocess the object patch for your model
        # input_for_your_model = preprocess_for_your_model(obj_img)
        # # Run your model on the object patch
        # your_model_prediction = your_model.predict(input_for_your_model)
        
        # Add text above the bounding box
        # cv2.putText(result.orig_img, f'Predict: {your_model_prediction}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Show the image
cv2.imshow("Image", result.orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()