from djitellopy import tello
import numpy as np
import cv2

frameWidth, frameHeight = 360, 240 # size of image

## Connect Tello and access camera
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

## Empty function
def empty(a):
    pass

## Create window to display HSV value trackbars
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 20, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 40, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 148, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 89, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

## Create window to display Canny threshold value trackbars
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 166, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 171, 255, empty)
cv2.createTrackbar("Area", "Parameters", 1750, 30000, empty)

while True:
    img = me.get_frame_read().frame ## store individual image from Tello
    img = cv2.resize(img, (frameWidth, frameHeight)) ## resize img to set width and height
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) ## convert image to HSV and store

    ## Store chosen HSV trackbar values
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min]) ## store lower bounds of HSV values
    upper = np.array([h_max, s_max, v_max]) ## store upper bounds of HSV values
    mask = cv2.inRange(imgHsv, lower, upper) ## create mask using lower and upper HSV range
    result = cv2.bitwise_and(img, img, mask=mask) ## keep only parts of image within mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) ## convert mask from grayscale to color
    print(f'[{h_min},{s_min},{v_min},{h_max},{s_max},{v_max}]') ## print HSV values

    imgBlur = cv2.GaussianBlur(result, (7, 7), 1) ## create blurred image to smooth for preprocessing
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY) ## convert blurred image from color to grayscale

    ## Store chosen Canny threshold trackbar values
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2) ## create Canny image detecting edges using threshold values
    kernel = np.ones((5, 5)) ## create 5x5 pixel kernel
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1) ## dilate the image using the kernel to increase edge sizes
    print(f'[{threshold1},{threshold2}]') ## print Canny threshold values

    hStack = np.hstack([img, mask, result]) ## create horizontal image stack of regular image, mask, and masked image
    cv2.imshow('Horizontal Stacking', hStack) ## display horizontal stacked images
    cv2.imshow('Image Dilation', imgDil) ## display final dilated image detecting edges

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()