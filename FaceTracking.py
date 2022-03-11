import cv2
import numpy as np
from djitellopy import tello
import time

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 25, 0)
time.sleep(3.5)

fbRange = [6200, 6800] ## face area range for drone to stay within
pid = [0.4, 0.4, 0] ## error formula parameters
pError = 0 ## error value
w, h = 360, 240 ## image dimensions

def findFace(img):
    faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml') ## file used to detect faces
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## convert img to grayscale
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8) ## detect faces using tested values

    myFaceListC = [] ## center point of faces
    myFaceListArea = [] ## area of faces

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) ## draw rectangle around face
        cx = x + w // 2 ## calculate horizontal center of face box
        cy = x + y // 2 ## calculate vertical center of face box
        area = w * h ## calculate area of face box
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED) ## draw circle as center point of face box
        myFaceListC.append([cx, cy]) ## add center point of face box
        myFaceListArea.append(area) ## add area of face box
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea)) ## store index of largest face box
        return img, [myFaceListC[i], myFaceListArea[i]] ## return center point and area of largest face box
    else:
        return img, [[0, 0], 0] ## return zeroes if no faces detected

def trackFace(info, w, pid, pError):
    area = info[1] ## store area from findFace
    x, y = info[0] ## store center points from findFace
    fb = 0 ## intialize fb movement to 0

    error = x - w // 2 ## calculate distance from center of face box to center of image
    speed = pid[0] * error + pid[1] * (error - pError) ## calculate amount of rotation needed based on center point error
    speed = int(np.clip(speed, -100, 100)) ## limit the speed of rotation between -100 and 100

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0 ## no movement needed if face box area is in range
    if area > fbRange[1]:
        fb = -20 ## move -20 backward if face box area is too large
    elif area < fbRange[0] and area != 0:
        fb = 20 ## move 20 forward if face box area is too small
    if x == 0:
        speed = 0 ## no movement if no center point
        error = 0 ## no error if no center point

    #print(speed, fb)

    me.send_rc_control(0, fb, 0, speed) ## send controls to drone
    return error

#cap = cv2.VideoCapture(0)

while True:
  #_, img = cap.read()
  img = me.get_frame_read().frame ## store individual image from Tello
  img = cv2.resize(img, (w, h)) ## resize img to set width and height
  img, info = findFace(img) ## call findFace function to find faces in image
  pError = trackFace(info, w, pid, pError) ## call trackFace function to determine position and drone movement
  #print('Center', info[0], 'Area', info[1])
  cv2.imshow('Output', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      me.land() ## stop and land drone if q key is pressed
      break