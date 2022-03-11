from djitellopy import tello
import numpy as np
import cv2
import cvlib as cv
import KeyPressModule as kp
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

## load the trained CNN model
model = load_model('Resources/GenderCNN/augmentedgender.hdf5')

w, h = 360, 240 ## size of image
classes = ['Woman', 'Man'] ## assign names to the gender labels

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

while True:
    img = me.get_frame_read().frame ## store individual image from Tello
    img = cv2.resize(img, (w, h)) ## resize img to set width and height

    face, confidence = cv.detect_face(img) ## detect and store faces from image

    ## loop through each face
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1] ## store starting x and y positions of face box
        (endX, endY) = f[2], f[3] ## store ending x and y positions of face box
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2) ## draw box around detected face
        face_crop = np.copy(img[startY:endY, startX:endX]) ## crop and store image of face by itself

        # go back to beginning of for loop if cropped face is too small
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96)) ## resize cropped face to input shape of CNN model
        face_crop = face_crop.astype("float") / 255.0 ## convert image to float and divide by 255 to normalize
        face_crop = img_to_array(face_crop) ## convert image to array
        face_crop = np.expand_dims(face_crop, axis=0) ## create new axis to reshape array

        conf = model.predict(face_crop)[0] ## use trained CNN model to predict label/gender of face picture
        idx = np.argmax(conf) ## store index of highest of two label values
        label = classes[idx] ## assign label name to predicted label
        label = "{}: {:.2f}%".format(label, conf[idx] * 100) ## format the label name and confidence percentage of prediction

        Y = startY - 10 if startY - 10 > 10 else startY + 10 ## calculate y position to place text

        cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2) ## write text of predicted label name

    cv2.imshow("Output", img) ## display image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()  ## stop and land drone if q key is pressed
        break