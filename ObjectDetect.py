import cv2
from djitellopy import tello
import KeyPressModule as kp
import time
import cvlib as cv
from cvlib.object_detection import draw_bbox

w, h = 360, 240 ## size of image
thres = 0.60 ## confidence threshold to detect object
global img ## Tello camera image

## Initialize Key Press Module
kp.init()
## Connect Tello and access camera
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

## Keyboard control of Tello
def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 30

    if kp.getKey('LEFT'): lr = -speed
    elif kp.getKey('RIGHT'): lr = speed

    if kp.getKey('UP'): fb = speed
    elif kp.getKey('DOWN'): fb = -speed

    if kp.getKey('w'): ud = speed
    elif kp.getKey('s'): ud = -speed

    if kp.getKey('a'): yv = -speed
    elif kp.getKey('d'): yv = speed

    if kp.getKey('q'): me.land(); time.sleep(3)
    if kp.getKey('e'): me.takeoff()

    if kp.getKey('z'):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
        time.sleep(0.3)

    return [lr, fb, ud, yv]

##### YOLO Implementation (Slower with Tello) #####
#while True:
  #img = me.get_frame_read().frame ## get keyboard button controls
  #img = cv2.resize(img, (w, h)) ## send keyboard controls to Tello
  #bbox, label, conf = cv.detect_common_objects(img) ## detect objects using OpenCV built-in YOLO object detector
  #img = draw_bbox(img, bbox, label, conf) ## draw bounding boxes around detected objects with labels
  #cv2.imshow('Output', img) ## display image
  #cv2.waitKey(1)

##### Mobilenet SSD Implementation (Faster with Tello) #####
classNames = [] ## create empty list to store names of detected objects
classFile = 'Resources/coco.names' ## file containing object class names that model can detect
with open(classFile, 'rt') as f:
  classNames = f.read().rstrip('\n').split('\n') ## read and store all of the classes stripping white space and line breaks

configPath = 'Resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' ## file containing configuration info for object detection model
weightsPath = 'Resources/frozen_inference_graph.pb' ## file containing weights for object detection model

net = cv2.dnn_DetectionModel(weightsPath, configPath) ## create object detector using pretrained OpenCV DNN model and files
net.setInputSize(320, 320) ## set input size parameter using tested/default values
net.setInputScale(1.0/127.5) ## set input scale parameter using tested/default values
net.setInputMean((127.5, 127.5, 127.5)) ## set input mean parameter using tested/default values
net.setInputSwapRB(True) ## set swap input and output channels parameter to true

while True:
    vals = getKeyboardInput() ## get keyboard button controls
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3]) ## send keyboard controls to Tello
    img = me.get_frame_read().frame ## store individual image from Tello
    img = cv2.resize(img, (w, h)) ## resize img to set width and height

    classIds, confs, bbox = net.detect(img, confThreshold=thres) ## use model detector on img with set confidence threshold for detecting objects
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2) ## draw bounding box of each detected object on img
            cv2.putText(img, classNames[classId-1].upper(),(box[0]+10,box[1]+30), ## write text of detected object's name on img
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img, str(round(confidence*100,2)),(box[0]+200,box[1]+30), ## write text of confidence percentage of detected object on img
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output", img) ## display image
    cv2.waitKey(1)