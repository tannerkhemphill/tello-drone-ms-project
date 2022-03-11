import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from importlib import reload
reload(keras.models)

w, h = 96, 96 ## image input size
lr = 0.001 ## learning rate for training
epoch = 30 ## training epochs
batch = 64 ## training batch size

## load the four arrays
x_train = np.load('Resources/GenderCNN/x_train.npy')
x_test = np.load('Resources/GenderCNN/x_test.npy')
y_train = np.load('Resources/GenderCNN/y_train.npy')
y_test = np.load('Resources/GenderCNN/y_test.npy')

## augment the image data to create more images for training
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

## create the CNN training model as a sequential model consisting of convolution and pooling layers with batch normalization and dropout for improved training
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(w,h,3))) ## set shape of input data for input layer
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) ## flatten the data for classification
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid')) ## final layer of two nodes to classify picture as man or woman

opt = keras.optimizers.Adam(learning_rate=lr, decay=lr/epoch) ## set training optimizer to Adam with learning rate and decay
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) ## set the parameters of the training model
model.summary()

gender = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch), steps_per_epoch=len(x_train) // batch, epochs=epoch, verbose=1, validation_data=(x_test, y_test)) ## train the model on the data

model.save('Resources/GenderCNN/augmentedgender.hdf5') ## save the trained model so it only has to be trained once

score = model.evaluate(x_test, y_test, verbose=1) ## evaluate the model based on the test data
print('\n', 'Test accuracy:', score[1]) ## print the accuracy of the evaluation
