import cv2
import numpy as np
from skimage import img_as_float
from skimage.segmentation import felzenszwalb
import tensorflow as tf

def selective_search(image, scale=100, sigma=0.5, min_size=20):
    # Convert the image to range 0 - 1
    if image.dtype != np.float64:
        image = img_as_float(image)
    # Perform Felzenszwalb segmentation
    segments = felzenszwalb(image, scale, sigma, min_size)
    

    # Initialize a list to hold region proposals
    region_proposals = []

    # Loop over unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):
        # Construct a mask for the segment
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out non-rectangular regions and small regions
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 4:  # Consider only rectangular regions
                (x, y, w, h) = cv2.boundingRect(approx)
                if w * h > 100:  # Filter out small regions based on size
                    region_proposals.append([x, y, x + w, y + h])

    # Merge overlapping boxes
    region_proposals = merge_boxes(region_proposals)

    return image, region_proposals
def merge_boxes(boxes, iou_threshold=0.5):
    merged_boxes = []
    while boxes:
        main_box = boxes.pop(0)
        boxes = [box for box in boxes if IoU(main_box, box) < iou_threshold]
        merged_boxes.append(main_box)
    return merged_boxes


def IoU(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xi1, yi1, xi2, yi2 = max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)
    intersection = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union = box1_area + box2_area - intersection

    return intersection / union
# Load the trained CNN model
model = tf.keras.models.load_model('new_fresh_model_signs.h5')

# Open the video file
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Apply the selective search function to the frame
    frame, proposals = selective_search(frame)
    

    # Define the threshold for the minimum proposal area
    area_threshold = 5000

    # Extract the proposal regions from the frame and resize
    proposals_regions = []
    proposals_to_evaluate = []
    for (x1, y1, x2, y2) in proposals:
        area = (x2 - x1) * (y2 - y1)
        if area > area_threshold:
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_CUBIC)
            proposals_regions.append(roi)
            proposals_to_evaluate.append((x1, y1, x2, y2))

    # Convert to numpy array and normalize
    proposals_regions = np.array(proposals_regions, dtype='float32') / 255.0

    # Make predictions on the proposal regions
    predictions = model.predict(proposals_regions)

    # Set a confidence threshold for detection
    confidence_threshold = 0.98

    # Draw bounding boxes on the detected traffic signs
    for idx, prediction in enumerate(predictions):
        class_index = np.argmax(prediction)
        confidence = prediction[class_index]
        if confidence > confidence_threshold:
            (x1, y1, x2, y2) = proposals_to_evaluate[idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Video', frame)
    
    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()