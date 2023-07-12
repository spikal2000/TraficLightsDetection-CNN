# from ultralytics import YOLO
# from PIL import Image
# import cv2

# model = YOLO(r"C:\Users\spika\Desktop\working yolo\last(1).pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source=r"C:\Users\spika\Desktop\working yolo\1.mp4")
# results = model.predict(source=r"C:\Users\spika\Desktop\working yolo", show=True) # Display preds. Accepts all YOLO predict arguments
# print(results.xywh[0])
# from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])

from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator

# model = YOLO('yolov8n.pt')
model = YOLO(r"C:\Users\spika\Desktop\working yolo\last(3).pt")
cap = cv2.VideoCapture(r"C:\Users\spika\Desktop\working yolo\g1.mp4")
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

            # Show detected object
            x1, y1, x2, y2 = [int(coord) for coord in b]
            detected_object = frame[y1:y2, x1:x2]
            cv2.imshow('Detected object', detected_object)

            
          
    frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', frame)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()



# for result in results:
#     # detection
#     result.boxes.xyxy   # box with xyxy format, (N, 4)
#     result.boxes.xywh   # box with xywh format, (N, 4)
#     result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
#     result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
#     result.boxes.conf   # confidence score, (N, 1)
#     result.boxes.cls    # cls, (N, 1)

#     # segmentation
#     result.masks.masks     # masks, (N, H, W)
#     result.masks.segments  # bounding coordinates of masks, List[segment] * N

#     # classification
#     result.probs     # cls prob, (num_class, )
