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
#me.send_rc_control(0, 0, 25, 0)
#time.sleep(2)


frameWidth, frameHeight = 640, 480 ## size of image
hsvVals = [37, 53, 36, 88, 180, 158] ## store HSV values as determined using ObjectFinder program
threshVals = [50, 226] ## store Canny threshold values as determined using ObjectFinder program
areaVal = 1750 ## set minimum area of object to track
deadZone = 100 ## set deadzone between segments of image
fbRange = [4000, 5000] ## object area range for drone to stay within

global imgContour
global dir ## direction to send drone

## Function to find object within image using contours
def getContours(img, imgContour):
    global dir
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
            cx = int(x + (w / 2))  # calculate horizontal center of object box
            cy = int(y + (h / 2))  # calculate vertical center of object box

            ## if object is too far left of horizontal center point of image
            if (cx < int(frameWidth / 2) - deadZone):
                cv2.putText(imgContour, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (0, int(frameHeight / 2 - deadZone)),
                              (int(frameWidth / 2) - deadZone, int(frameHeight / 2) + deadZone), (0, 0, 255),
                              cv2.FILLED)
                dir = 1

            ## if object is too far right of horizontal center point of image
            elif (cx > int(frameWidth / 2) + deadZone):
                cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 + deadZone), int(frameHeight / 2 - deadZone)),
                              (frameWidth, int(frameHeight / 2) + deadZone), (0, 0, 255), cv2.FILLED)
                dir = 2

            ## if object is too far above vertical center point of image
            elif (cy < int(frameHeight / 2) - deadZone):
                cv2.putText(imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZone), 0),
                              (int(frameWidth / 2 + deadZone), int(frameHeight / 2) - deadZone), (0, 0, 255),
                              cv2.FILLED)
                dir = 3

            ## if object is too far below vertical center point of image
            elif (cy > int(frameHeight / 2) + deadZone):
                cv2.putText(imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZone), int(frameHeight / 2) + deadZone),
                              (int(frameWidth / 2 + deadZone), frameHeight), (0, 0, 255), cv2.FILLED)
                dir = 4

            ## if object is too close based on area
            elif area > fbRange[1]:
                cv2.putText(imgContour, " GO BACKWARD ", (620, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                dir = 5

            ## if object is too far based on area
            elif area < fbRange[0] and area != 0:
                cv2.putText(imgContour, " GO FORWARD ", (620, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                dir = 6

            ## if object is close enough to center of image
            else:
                dir = 0

            cv2.line(imgContour, (int(frameWidth / 2), int(frameHeight / 2)), (cx, cy), (0, 0, 255), 3) ## draw line from center point of object to center point of image
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5) ## draw rectangle around object
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2) ## write text of approximated perimeter of object contour
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2) ## write text of area of object contour
            cv2.putText(imgContour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (0, 255, 0), 2) ## write text of bounding box's x and y

        ## if object area is
        else:
            dir = 0

## Function to draw lines and center point on image dividing it up into square regions
def display(img):
    cv2.line(img, (int(frameWidth / 2) - deadZone, 0), (int(frameWidth / 2) - deadZone, frameHeight), (255, 255, 0), 3)
    cv2.line(img, (int(frameWidth / 2) + deadZone, 0), (int(frameWidth / 2) + deadZone, frameHeight), (255, 255, 0), 3)
    cv2.circle(img, (int(frameWidth / 2), int(frameHeight / 2)), 5, (0, 0, 255), 5)
    cv2.line(img, (0, int(frameHeight / 2) - deadZone), (frameWidth, int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)

while True:
    img = me.get_frame_read().frame ## store individual image from Tello
    img = cv2.resize(img, (frameWidth, frameHeight)) ## resize img to set width and height
    imgContour = img.copy() ## copy image and store
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) ## convert image to HSV

    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]]) ## store lower bounds of HSV values
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]]) ## store upper bounds of HSV values
    mask = cv2.inRange(imgHsv, lower, upper) ## create mask using lower and upper HSV range
    result = cv2.bitwise_and(img, img, mask=mask) ## keep only parts of image within mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) ## convert mask from grayscale to color

    imgBlur = cv2.GaussianBlur(result, (7, 7), 1) ## create blurred image to smooth for preprocessing
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY) ## convert blurred image from color to grayscale
    threshold1 = threshVals[0] ## store lower bounds of Canny threshold values
    threshold2 = threshVals[1] ## store upper bounds of Canny threshold values
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2) ## create Canny image detecting edges using threshold values
    kernel = np.ones((5, 5)) ## create 5x5 pixel kernel
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1) ## dilate the image using the kernel to increase edge sizes
    getContours(imgDil, imgContour) ## call getContours to find object and determine position
    display(imgContour) ## call display function to draw square regions

    ################# FLIGHT
    if dir == 1:
        me.yaw_velocity = -60
    elif dir == 2:
        me.yaw_velocity = 60
    elif dir == 3:
        me.up_down_velocity = 60
    elif dir == 4:
        me.up_down_velocity = -60
    elif dir == 5:
        me.for_back_velocity = -60
    elif dir == 5:
        me.for_back_velocity = 60
    else:
        me.left_right_velocity = 0
        me.for_back_velocity = 0
        me.up_down_velocity = 0
        me.yaw_velocity = 0
    # SEND VELOCITY VALUES TO TELLO
    #if me.send_rc_control:
        #me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
    print(dir)

    cv2.imshow('Mask', mask) ## display masked image
    cv2.imshow('Image Contour', imgContour) ## display image divided into regions

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land() ## stop and land drone if q key is pressed
        break

cv2.destroyAllWindows()