import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500

    model = Sequential()
    # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(
        imgdimension[0], imgdimension[1], 1), activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dropout(0.5))
    model.add(Dense(numberofdirectories, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


path = 'myData'
test_ratio = 0.2
val_ratio = 0.2
imgdimension = (32, 32, 3)
labelFile = 'labels.csv'  # file with all names of classes
batch_size_val = 50  # how many to process together
steps_per_epoch_val = 2000
epochs_val = 10

images = []
Classno = []
directories = os.listdir(path)
numberofdirectories = len(directories)
print("Total number of Classes Detected", numberofdirectories)
for x in range(0, numberofdirectories):
    imagelist = os.listdir(path + "/" + str(x))
    for y in imagelist:
        currentimage = cv2.imread(path + "/" + str(x) + "/" + y)
        currentimage = cv2.resize(
            currentimage, (imgdimension[0], imgdimension[1]))
        images.append(currentimage)
        Classno.append(x)
images = np.array(images)
Classno = np.array(Classno)


# print(Classno.shape)
# print(images.shape)


# Spliting the data

X_train, X_test, Y_train, Y_test = train_test_split(
    images, Classno, test_size=test_ratio)
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=val_ratio)

noofsamples = []
for i in range(10):
    noofsamples.append(len(np.where(Y_train == i)[0]))
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

print(X_train.shape)
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(
    X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(
    X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(X_train)


Y_train = to_categorical(Y_train, numberofdirectories)
Y_test = to_categorical(Y_test, numberofdirectories)
Y_validation = to_categorical(Y_validation, numberofdirectories)


model = myModel()
print(model.summary())

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val, validation_data=(X_validation, Y_validation), shuffle=1)
pickle_out = open("model.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()


cv2.waitKey(0)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
