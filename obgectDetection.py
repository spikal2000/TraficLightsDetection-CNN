import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#camera
cap = cv2.VideoCapture(0)

#model
#model = load_model('model.h5')

while True:
    ret,frame = cap.read()

    # Detecting objects with yolo
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #loop through the detected objects
    for out in outs:
        for detection in out:   
            scotres = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                #object detected
                center_x = int(detection[0]* frame.shape[1])
                center_y = int(detection[1]* frame.shape[0])
                w = int(detection[2]* frame.shape[1])
                h = int(detection[3]* frame.shape[0])
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                #draw a rectangle
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

                #Crop, process the region for cnn
                #roi = pre_process(frame[y:y+h, x:x+w])

                #predict the class
                #pred = model.predict(roi)

                #put text
                #cv2.putText(frame, pred, (x,y), font, 1, (0,0,255), 2)
    cv2.imshow('frame', frame)

    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()