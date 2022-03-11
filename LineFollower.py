import cv2
import numpy as np
from djitellopy import tello

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
#me.takeoff()

#cap = cv2.VideoCapture(0)

hsvVals = [0, 0, 188, 179, 33, 245] # values used to isolate path based on color from testing
sensors = 3 ## number of ways image is split
threshold = 0.2 ## percent of total pixels used for numbering pixels 0 or 1
width, height = 480, 360 ## dimensions of image

sensitivity = 3 ## if number is high less sensitive adjustments to movements

weights = [-25, -15, 0, 15, 25] ## rotation weights depending on senOut values
fSpeed = 15 ## constant forward speed of drone
curve = 0 ## rotation value

def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) ## convert img to hsv for color thresholding
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]]) ## lower limit of hsvVals for threshold
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]]) ## upper limit of hsvVals for threshold
    mask = cv2.inRange(hsv, lower, upper) ## variable containing part of img within lower and upper ranges of threshold
    return mask

def getContours(imgThres, img):
    cx = 0
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) ## find edges within imgThres
    if len(contours) != 0:
        biggest = max(contours, key = cv2.contourArea) ## store largest contour/region by area to represent path
        x, y, w, h = cv2.boundingRect(biggest) ## store dimensions of largest region/path
        cx = x + w//2 ## calculate horizontal center of imgThres
        cy = y + h//2 ## calculate vertical center of imgThres
        cv2.drawContours(img, contours, -1, (255, 0, 255), 7) ## draw edges of path on img
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED) ## draw center point of path on img

    return cx

def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors) ## split imgThres into 3 sensor images
    totalPixels = (img.shape[1]//sensors) * img.shape[0] ## calculate total pixels in each sensor by multiplying divided width of image by height of image
    senOut = []
    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im) ## calculate total number of pixels within each sensor
        if pixelCount > threshold*totalPixels:
            senOut.append(1) ## add 1 for sensor if detected number of pixels are greater than threshold percentage
        else:
            senOut.append(0) ## add 0 for sensor if detected number of pixels are less than threshold percentage
        #cv2.imshow(str(x), im)
    #print(senOut)
    return senOut

def sendCommands(senOut, cx):
    global curve

    ## Translation
    lr = (cx - width//2)//sensitivity ## calculate left-right movement value based on center point with sensitivity included
    lr = int(np.clip(lr, -10, 10)) ## limit movement range between -10 and 10
    if lr < 2 and lr > -2: lr = 0 ## prevent continuous movement when numbers are insignificant

    ## Rotation
    if senOut ==   [1, 0, 0]: curve = weights[0] ## rotate drone -25 to left
    elif senOut == [1, 1, 0]: curve = weights[1] ## rotate drone -15 to left
    elif senOut == [0, 1, 0]: curve = weights[2] ## rotate drone straight at 0
    elif senOut == [0, 1, 1]: curve = weights[3] ## rotate drone -15 to right
    elif senOut == [0, 0, 1]: curve = weights[4] ## rotate drone -25 to right

    elif senOut == [0, 0, 0]: curve = weights[2] ## do nothing when drone does not detect path
    elif senOut == [1, 1, 1]: curve = weights[2] ## do nothing when drone detects path on all sensors
    elif senOut == [1, 0, 1]: curve = weights[2] ## do nothing when drone detects split path

    #me.send_rc_control(lr, fSpeed, 0, curve)

while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    #img = cv2.flip(img, 0)

    imgThres = thresholding(img) ## create thresholded image of img contents
    cx = getContours(imgThres, img) ## for translation/left-right movement of drone
    senOut = getSensorOutput(imgThres, sensors) ## for rotation of drone
    sendCommands(senOut, cx) ## send commands to move drone according to cx and senOut values

    cv2.imshow('Output', img) ## display regular video output
    cv2.imshow('Path', imgThres) ## display thresholded video output
    cv2.waitKey(1)