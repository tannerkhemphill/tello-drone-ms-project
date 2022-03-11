from djitellopy import tello
from time import sleep
import KeyPressModule as kp
import numpy as np
import cv2
import math

### Parameters ###
fSpeed = 117/10 ## actual measured forward speed in cm/s (supposed to be 15 cm/s)
aSpeed = 360/10 ## actual measured angular speed in deg/s
interval = 0.25 ## shorten interval to calculate positioning
dInterval = fSpeed*interval ## calculate distance position with interval
aInterval = aSpeed*interval ## calculate angular position with interval
##################

x, y = 500, 500 ## drone starting position points
a = 0 ## drone angle variable
yaw = 0 ## drone rotation variable
points = [(0, 0), (0, 0)] ## point position of drone

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0 ## drone movement variables
    speed = 15 ## speed used for calculations
    aspeed = 50 ## speed used for calculations
    d = 0 ## drone distance variable (reset each function call to determine distance from previous point)
    global x, y, a, yaw

    if kp.getKey('LEFT'):
        lr = -speed
        d = dInterval ## update distance on map to the left
        a = -180 ## update angle on map to the left

    elif kp.getKey('RIGHT'):
        lr = speed
        d = -dInterval ## update distance on map to the right
        a = 180 ## update angle on map to the right

    if kp.getKey('UP'):
        fb = speed
        d = dInterval ## update distance on map forward/"up"
        a = 270 ## update angle on map foward/"up"

    elif kp.getKey('DOWN'):
        fb = -speed
        d = -dInterval ## update distance on map backward/"down"
        a = -90 ## update angle on map backward/"down"

    if kp.getKey('w'):
        ud = speed

    elif kp.getKey('s'):
        ud = -speed

    if kp.getKey('a'):
        yv = -aspeed
        yaw -= aInterval ## update angular position rotating left

    elif kp.getKey('d'):
        yv = aspeed
        yaw += aInterval ## update angular position rotating right

    if kp.getKey('q'): me.land(); sleep(3)
    if kp.getKey('e'): me.takeoff()

    sleep(interval) # sync drone movement to mapping interval
    a += yaw ## update angle
    x += int(d * math.cos(math.radians(a))) ## calculate and update x position including angular movement for map
    y += int(d * math.sin(math.radians(a))) ## calculate and update y position including angular movement for map

    return [lr, fb, ud, yv, x, y]

def drawPoints(img, points):
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED) ## draw every point
    cv2.circle(img, points[-1], 8, (0, 255, 0), cv2.FILLED) ## draw most recent point different color
    cv2.putText(img, f'({(points[-1][0] - 500)/100}, {-(points[-1][1] - 500)/100})m',
                (points[-1][0] + 10, points[-1][1] + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1) ## write coordinates

while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))

    map = np.zeros((1000, 1000, 3), np.uint8) ## scale map to 1000x1000 with 3 color channels and 0-255 possible values
    if (points[-1][0] != vals[4] or points[-1][1] != vals[5]):
        points.append((vals[4], vals[5])) ## add current drone position if it has changed from previous position
    drawPoints(map, points) ## draw point of drone position on map
    cv2.imshow('Image', img)
    cv2.imshow('Map', map)
    cv2.waitKey(1)
