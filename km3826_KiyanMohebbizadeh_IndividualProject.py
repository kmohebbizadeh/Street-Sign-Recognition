# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, AveragePooling2D

# load the data
train = loadmat('./SVHN/format2/train_32x32.mat')
test = loadmat('./SVHN/format2/test_32x32.mat')

# set the data into the respective arrays
X_train = train['X']
y_train = train['y']

X_test = test['X']
y_test = test['y']

# rearrange the shape of the data for usage in the model (essentially make each row of the dataset an image instead of every column)
X_train = np.rollaxis(X_train, axis=-1)
X_test = np.rollaxis(X_test, axis=-1)

# encode the y variable (best practice for multi-class categorical)
y_train = to_categorical(y_train, 11)
y_test = to_categorical(y_test, 11)

# create a small validation set. 90:10 split. we want to stratify here to make sure we get some of every category in the validation set
X_dev, X_val, y_dev, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# print out the shapes to make sure the preprocessing worked and is usable
print('X Development shape: ', X_dev.shape)
print('X Validation shape: ', X_val.shape)
print('X Train shape: ', X_train.shape)
print('X Test shape: ', X_test.shape)
print('y Development shape (encoded): ', y_dev.shape)
print('y Validation shape (encoded): ', y_val.shape)
print('y train shape (encoded): ', y_train.shape)
print('y test shape (encoded): ', y_test.shape)

'''
Creating the model:
- this model is adapted and modified from LeNet 5
- I use the elu activation function and decided this by experimenting with other activation functions.
- I use softmax for the last layer for best decision making.
- conv2d in this case I am using a convolutional kernel on a 2 dimensional image with padding. I set the kernel to be small 3x3 because the
images are relatively clean so a detailed scan of the image is acceptable, the input shape is the size of the image and strides is how many pixels
are jumped at each application of the kernel, filters is the dimensions of the output from this layer.
- average pooling is a type of spatial reduction or dimensionality reduction. this works by taking the average of the pool size across the image
- conv2d similar to the first convolutional layer, but input shape is no longer required and no padding for this layer more filters for detailed output
- average pooling same as above
- conv2d similar to other convolutional layers, just a larger filter set for more detailed output
- flatten makes a 1 dimensional array for dense layers
- dropout removes 10% of the nodes at random to avoid over-fitting (applicable for the 2 large Dense layers)
- Dense layers are used in neural networks, here we reduce number of nodes at each layer until a decision is made. This type of model 
iterates through the features and narrows down which aspects are most important until a decision is made
CNN Compilation
- I use the adamax optimizer decided by experimentation with other optimizers
- I use categorical cross entropy since we encoded the y array for the loss value
- I want to tune for accuracy since that is the goal of the assignment.
'''

cnn = Sequential()
cnn.add(Conv2D(filters=6, kernel_size=(3, 3), activation='elu', strides=(1, 1), padding='same', input_shape=(32, 32, 3)))
cnn.add(AveragePooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=16, kernel_size=(3, 3), activation='elu', strides=(1, 1), padding='valid'))
cnn.add(AveragePooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=120, kernel_size=(3, 3), activation='elu', strides=(1, 1), padding='valid'))
cnn.add(Flatten())
cnn.add(Dropout(rate=0.1, seed=42))
cnn.add(Dense(units=256, activation='elu'))
cnn.add(Dropout(rate=0.1, seed=42))
cnn.add(Dense(units=128, activation='elu'))
cnn.add(Dropout(rate=0.1, seed=42))
cnn.add(Dense(units=64, activation='elu'))
cnn.add(Dense(units=32, activation='elu'))
cnn.add(Dense(units=16, activation='elu'))
cnn.add(Dense(units=11, activation='softmax'))

cnn.compile(optimizer="Adamax", loss="categorical_crossentropy", metrics=["accuracy"])

cnn.summary()

'''
I use the Image Data Generator to augment the images slightly for better training of the model. 
each epoch gets a slightly different set of images with slight rotation, shifting, zooming in or out. 

I only augment the development data not the validation or test data. This is standard practice for training models on augmented data.

I set all the ranges relatively low for a few reasons, one the images are already clean and cropped. This means that 
big changes may exclude important parts of the image. Second we are looking for a "jitter" effect on the images, just slight
changes so that the epochs arent trained on the same exact data set creating an over fit model.

- rotation range is how many degrees the image can be rotated to either side
- width shift range is how much the image can be moved from side to side
- height shift range is how much the image can be moved up and down 
- shear range is how much can an image can be distorted either on the x or y axis
- zoom range is how much can be trimmed from around the image

I set the batch size to be very small to avoid over fitting. I set the epochs to 15 by exploration and tuning not allowing for overfitting, 
but enough to fully optimize the CNN.
'''

dev_idg = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=.075,
    height_shift_range=.075,
    shear_range=.1,
    zoom_range=.1)

val_idg = ImageDataGenerator()

dev_idg = dev_idg.flow(x=X_dev, y=y_dev, batch_size=5, seed=42)
val_idg = val_idg.flow(x=X_val, y=y_val, batch_size=5, seed=42)

# fit and evaluate the model
history_callback = cnn.fit(dev_idg, epochs=15, validation_data=val_idg, verbose=1)
score = cnn.evaluate(X_test, y_test, verbose=1)
print('Accuracy:', score[1])

# charts for evaluation
plt.figure(figsize=(10, 10))
hist = pd.DataFrame(history_callback.history)
plt.plot(hist.index, hist["loss"], label='train loss')
plt.plot(hist.index, hist["val_loss"], label='validation loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.savefig("loss.png")

plt.figure(figsize=(10, 10))
hist = pd.DataFrame(history_callback.history)
plt.plot(hist.index, hist["accuracy"], label='accuracy')
plt.plot(hist.index, hist["val_accuracy"], label='validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend()
plt.savefig("accuracy.png")
