## Author          : Jolo Tolentino
## Project Name    : Yolo Prototype Structure
## Project Started : February 1,2022



### Detecor.py uses the YOLO object detection model by default but can be changed

import cv2 
import numpy as np

## adjust Threshold Value According to Desired Specification
Threshold = 0.4

## Implicit Location Addresses
Models_Path =  '.\Config' 
Yolo_Weights,Yolo_CFG,Labels_Path = Models_Path+'\Yolo.weights', Models_Path+'\Yolo.cfg' , Models_Path + "\coco.names"

# Labels and Colors have the same Length so colors is an array (same length as Labels) of BGR (OPENCV uses BGR) Tupples wi
LABELS = open(Labels_Path).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


## Loading the configuration Files of YOLO into Opencv
Yolo = cv2.dnn.readNet(Yolo_Weights, Yolo_CFG) 

# Loading 
All_Layer_Names = Yolo.getLayerNames()


Necessary_Layers = [All_Layer_Names[i - 1] for i in Yolo.getUnconnectedOutLayers()]
cam = cv2.VideoCapture(1)

while True: 
    _,Video_Feed = cam.read()
    Yolo_Video_Feed = Video_Feed.copy()
    Height,Width = Video_Feed.shape[0],Video_Feed.shape[1]
    
    ##we covnvert the video frame into a BLOB object for Yolo to understand. BLOB is a uniform standard scaling for the YOLO model to understand our input video frame 
    Video_Blob = cv2.dnn.blobFromImage(Video_Feed,1/255,(416,416))
    Yolo.setInput(Video_Blob)


    Yolo_Predictions = Yolo.forward(Necessary_Layers)
    
    Boxes, Confidences, Classification_IDs = [], [], []

    for Predictions in Yolo_Predictions:
        for objects in Predictions:
            ## each object is an array 
            scores = objects[5:]  ## Depending on the number of labels that would be the entire size of the scores array 
            Classification =  np.argmax(scores) ## get the highest scoring index
            Confidence = scores[Classification] ##acces the confidence score via the index we just extracted from the scores value (arg max ) 


            if Confidence>Threshold:
                box = objects[:4]*np.array([Width,Height,Width,Height]) #SCALES the YOLO PREDICTIONS TO CAMERA FEED STANDARDS
                (CenterX,CenterY,Width,Height) = box.astype('int')

                XMin = int(CenterX - Width//2) 
                YMin = int(CenterY - Height//2) 

                ######## Store for Later usage
                Boxes.append([XMin,YMin,int(Width),int(Height)]) 
                Confidences.append(float(Confidence))
                Classification_IDs.append(Classification)
    
    # reduces redundant boxes
    Indexes = cv2.dnn.NMSBoxes(Boxes,Confidences,Threshold,Threshold)

    
    if len(Indexes) > 0:
        for i in Indexes.flatten():       
            (x, y) = (Boxes[i][0], Boxes[i][1])
            (w, h) = (Boxes[i][2], Boxes[i][3])          
            ## checking corresponding color for Class Predicted
            color = [int(c) for c in COLORS[Classification_IDs[i]]]
            ## Drawing Information into copied Video Frame 
            cv2.rectangle(Yolo_Video_Feed, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}".format(LABELS[Classification_IDs[i]], Confidences[i])
            cv2.putText(Yolo_Video_Feed, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

    cv2.imshow("YOLO",Yolo_Video_Feed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
