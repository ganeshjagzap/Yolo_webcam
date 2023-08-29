from ultralytics import YOLO
import cv2
import cvzone
import math

cap=cv2.VideoCapture(0)

# yolo pre-trained model with nano weight
model=YOLO('../weights/yolov8n.pt')

# Coco dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    suc,img=cap.read()
    results=model(img,stream=True) # stream=True is recommended to use because it makes use of generators
    for i in results:
        bboxes = i.boxes
        for j in bboxes:
            # open cv method or cv2 method
            '''
            x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            '''
            # cvzone method- more fancier bboxes
            x1, y1, x2, y2 = j.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width, height = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, width, height), colorC=(255, 0, 200))

            confid = math.ceil((j.conf[0] * 100)) / 100
            label = int(j.cls[0])

            cvzone.putTextRect(img, f'{classNames[label]} {confid}', (max(0, x1), max(30, y1)), scale=2,
                               thickness=1)



    cv2.imshow('image',img)
    cv2.waitKey(1)