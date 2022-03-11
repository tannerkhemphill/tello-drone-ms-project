import cv2
from djitellopy import tello
import MeasureFinder as mf

## Connect Tello and access camera
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

scale = 3 ## image scaling parameter
wP = 210 *scale ## scaled width of image
hP= 297 *scale ## scaled height of image

while True:
    img = me.get_frame_read().frame ## store individual image from Tello

    imgContours, conts = mf.getContours(img, minArea=50000, filter=4) # get contours in image above minimum area

    if len(conts) != 0:
        biggest = conts[0][2] ## get biggest contour
        # print(biggest)
        imgWarp = mf.warpImg(img, biggest, wP, hP) ## warp image to size of background
        imgContours2, conts2 = mf.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False) ## get contours in warped image
        if len(conts) != 0:
            ## loop through contours
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2) ## draw biggest contour
                nPoints = mf.reorder(obj[2]) ## reorder points of biggest contour
                nW = round((mf.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1) ## calculate width of contour
                nH = round((mf.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1) ## calculate heigth of contour
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05) ## draw arrowed line of width
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05) ## draw arrowed line of height
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2) ## write text of width
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2) ## write text of height
        cv2.imshow('A4', imgContours2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5) ## resize image
    cv2.imshow('Output', img) ## display image
    cv2.waitKey(1)