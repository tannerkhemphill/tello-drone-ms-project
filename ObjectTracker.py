from djitellopy import tello
import numpy as np
import cv2
import time

## Connect Tello and access camera
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

## set Tello to takeoff and rise up to level of object
me.takeoff()
me.send_rc_control(0, 0, 25, 0)
time.sleep(1)

frameWidth, frameHeight = 640, 480 ## size of image
hsvVals = [37, 53, 36, 88, 180, 158] ## store HSV values as determined using ObjectFinder program
threshVals = [50, 226] ## store Canny threshold values as determined using ObjectFinder program
areaVal = 1000 ## set minimum area of object to track
deadZone = 100 ## set deadzone between segments of image
fbRange = [2000, 4000] ## object area range for drone to stay within
pid = [0.4, 0.4, 0]
pError = 0

global imgContour
global dir ## direction to send drone

## Function to find object within image using contours
def getContours(img, imgContour, pid, pError):
    fb = 0 ## intialize fb movement to 0
    ud = 0 ## intialize ud movement to 0
    yaw = 0  ## intialize yaw movement to 0
    error = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) ## find and store edges/contours of image
    ## Loop through every contour
    for cnt in contours:
        area = cv2.contourArea(cnt) ## calculate area of contour
        areaMin = areaVal ## set minimum area to chosen value
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7) ## draw contour on imgContour if area is above minimum value
            peri = cv2.arcLength(cnt, True) ## calculate perimeter of contour
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) ## approximate the overall contour to ignore minor variations from perimeter
            x, y, w, h = cv2.boundingRect(approx) ## retrieve x, y, width, and height values from bounding rectangle of contour
            cx = int(x + (w / 2))  ## calculate horizontal center of object box
            cy = int(y + (h / 2))  ## calculate vertical center of object box

            error = cx - frameWidth // 2  ## calculate distance from center of face box to center of image
            yaw = pid[0] * error + pid[1] * (error - pError)  ## calculate amount of rotation needed based on center point error
            yaw = int(np.clip(yaw, -100, 100))  ## limit the speed of rotation between -100 and 100

            if area > fbRange[0] and area < fbRange[1]:
                fb = 0  ## no movement needed if object box area is in range
            if area > fbRange[1]:
                fb = -20  ## move -20 backward if object box area is too large
            elif area < fbRange[0] and area != 0:
                fb = 20  ## move 20 forward if object box area is too small
            if (cy < int(frameHeight / 2)):
                ud = 20 ## move 20 up if object center is too high
            elif (cy > int(frameHeight / 2)):
                ud = -20 ## move 20 up if object center is too low
        print(area)

    me.send_rc_control(0, fb, ud, yaw) ## send controls to drone
    return error

while True:
    img = me.get_frame_read().frame  ## store individual image from Tello
    img = cv2.resize(img, (frameWidth, frameHeight))  ## resize img to set width and height
    imgContour = img.copy()  ## copy image and store
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  ## convert image to HSV

    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])  ## store lower bounds of HSV values
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])  ## store upper bounds of HSV values
    mask = cv2.inRange(imgHsv, lower, upper)  ## create mask using lower and upper HSV range
    result = cv2.bitwise_and(img, img, mask=mask)  ## keep only parts of image within mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  ## convert mask from grayscale to color

    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)  ## create blurred image to smooth for preprocessing
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)  ## convert blurred image from color to grayscale
    threshold1 = threshVals[0] ## store lower bounds of Canny threshold values
    threshold2 = threshVals[1] ## store upper bounds of Canny threshold values
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2) ## create Canny image detecting edges using threshold values
    kernel = np.ones((5, 5))  ## create 5x5 pixel kernel
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)  ## dilate the image using the kernel to increase edge sizes
    pError = getContours(imgDil, imgContour, pid, pError) ## call getContours function to find object and determine position and drone movement

    cv2.imshow('Mask', mask)  ## display masked image
    cv2.imshow('Image Contour', imgContour)  ## display image divided into regions

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land() ## stop and land drone if q key is pressed
        break

cv2.destroyAllWindows()