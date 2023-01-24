import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2DTranspose
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from PIL import Image

from matplotlib import pyplot as plt

def default_categorical(input_shape=(30, 80, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop, l4_stride=2)
    x = Dense(100, activation='relu', name="dense_1")(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name="dense_2")(x)
    x = Dropout(drop)(x)
    # Categorical output of the angle into 15 bins
    probs = Dense(2, activation='softmax', name='output_probs')(x)
    
    model = Model(inputs=[img_in], outputs=[probs])
    return model

def core_cnn_layers(img_in, drop, l4_stride=1):
    """
    Returns the core CNN layers that are shared among the different models,
    like linear, imu, behavioural

    :param img_in:          input layer of network
    :param drop:            dropout rate
    :param l4_stride:       4-th layer stride, default 1
    :return:                stack of CNN layers
    """
    x = img_in
    x = conv2d(16, 5, (2,3), 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(24, 5, (2,2), 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, (2,2), 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(44, 3, (1,2), 4)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x

def conv2d(filters, kernel, strides, layer_num, activation='relu'):
    """
    Helper function to create a standard valid-padded convolutional layer
    with square kernel and strides and unified naming convention

    :param filters:     channel dimension of the layer
    :param kernel:      creates (kernel, kernel) kernel matrix dimension
    :param strides:     creates (strides, strides) stride
    :param layer_num:   used in labelling the layer
    :param activation:  activation, defaults to relu
    :return:            tf.keras Convolution2D layer
    """
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides[0], strides[1]),
                         activation=activation,
                         name='conv2d_' + str(layer_num))

#Load data
images = []
labels = []

for tub in os.listdir('data_yk_positive'):
    for image_file in os.listdir('data_yk_positive/'+tub+'/images'):
        img = Image.open(f'data_yk_positive/{tub}/images/{image_file}')
        img = np.asarray(img)
        img = img[60:,20:-20,:]
        images.append(img)
        labels.append(1)
        

for tub in os.listdir('data_yk_negative'):
    for image_file in os.listdir('data_yk_negative/'+tub+'/images'):
        img = Image.open(f'data_yk_negative/{tub}/images/{image_file}')
        img = np.asarray(img)
        img = img[60:,20:-20,:]
        images.append(img)
        labels.append(0)

for i, image_file in enumerate(os.listdir('data3/tub_1_22-11-05/images')):
        img = Image.open(f'data3/tub_1_22-11-05/images/{image_file}')
        img = np.asarray(img)
        img = img[60:,20:-20,:]
        images.append(img)
        labels.append(0)

images = np.array(images)
images = images/255
labels = np.array(labels)

order = np.arange(labels.shape[0])
np.random.shuffle(order)

images = images[order]
labels = labels[order]

split = int(labels.shape[0]*0.85)
images_train = images[:split]
labels_train = labels[:split]

images_val = images[split:]
labels_val = labels[split:]

labels_train = to_categorical(labels_train, num_classes=2)
labels_val = to_categorical(labels_val, num_classes=2)

input_shape=(60, 120, 3)
model = default_categorical(input_shape)

model.compile(optimizer="rmsprop", metrics=['accuracy'],
                           loss='categorical_crossentropy')

history = model.fit(
    images_train,
    labels_train,
    batch_size=2048,
    epochs=55,
    class_weight={0:1, 1:5},
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(images_val, labels_val),
)

model.save("model_yk.h5")

plt.ion()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.ioff()
plt.savefig("history_yk.png")
plt.show()