import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('new_fresh_model_signs.h5')

# Load image
image_path = "20.jpg"
img = cv2.imread(image_path)

# Initialize OpenCV's selective search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
rects = ss.process()

proposals = []
boxes = []

# Iterate over the region proposals
for x, y, w, h in rects[:200]:  # Limiting to 2000 region proposals
    # Extract the region from the original image
    roi = img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_CUBIC)
    
    proposals.append(roi)
    boxes.append((x, y, w, h))

proposals = np.array(proposals)
predictions = model.predict(proposals)
print(predictions)
threshold = 0.999
for pred, (x, y, w, h) in zip(predictions, boxes):
    max_pred = np.max(pred)
    if np.argmax(pred) == 1 and max_pred > threshold:  # If the proposal is classified as a traffic sign
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Image with ROIs', img)
cv2.waitKey(0)
