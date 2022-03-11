import cv2
import numpy as np

## Function to find object in an image using contours
def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## convert image to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) ## create blurred image to smooth for preprocessing
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1]) ## create Canny image detecting edges using threshold values for preprocessing
    kernel = np.ones((5, 5)) ## create 5x5 pixel kernel
    imgDil = cv2.dilate(imgCanny, kernel, iterations=3) ## dilate the image using the kernel to increase edge sizes for preprocessing
    imgThre = cv2.erode(imgDil, kernel, iterations=2) ## erode image using the kernel to decrease edge sizes for preprocessing
    if showCanny: cv2.imshow('Canny', imgThre) ## display image
    contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) ## find and store edges/contours of image
    finalCountours = [] ## create list to store final contours
    # loop through each contour
    for i in contours:
        area = cv2.contourArea(i) ## calculate area of contour
        if area > minArea:
            peri = cv2.arcLength(i, True) ## calculate perimeter of contour
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) ## approximate the overall contour to ignore minor variations from perimeter
            bbox = cv2.boundingRect(approx) ## draw bounding box around contour
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i]) ## add contour to list of final contours if length matches filter parameter
            else:
                finalCountours.append([len(approx), area, approx, bbox, i]) ## add contour to list of final contours if filter parameter is zero
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True) ## sort contours from largest to smallest
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3) ## draw contours on image
    return img, finalCountours

## Function to reorder points of object to consistent order for calculation
def reorder(myPoints):
    # print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints) ## create numpy array of zeros in same shape as points array
    myPoints = myPoints.reshape((4, 2)) ## reshape array for rectangles
    add = myPoints.sum(1) ## add the points
    myPointsNew[0] = myPoints[np.argmin(add)] ## set first point as minimum of sum
    myPointsNew[3] = myPoints[np.argmax(add)] ## set last point as maximum of sum
    diff = np.diff(myPoints, axis=1) ## subtract the points
    myPointsNew[1] = myPoints[np.argmin(diff)] ## set second point as minimum of difference
    myPointsNew[2] = myPoints[np.argmax(diff)] ## set third point as maximum of difference
    return myPointsNew

## Function to warp an image to dimensions of uniform background
def warpImg(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points) ## reorder the points
    pts1 = np.float32(points) ## convert points to float
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) ## convert second set of points to float using image dimensions
    matrix = cv2.getPerspectiveTransform(pts1, pts2) ## transform the first set of points to second set
    imgWarp = cv2.warpPerspective(img, matrix, (w, h)) ## warp the image to fill with background
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad] ## add pad to image for calculations
    return imgWarp

## Function to calculate the size of an object using points
def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5