import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


num_classes = 10
y_train = to_categorical(y_train, num_classes = num_classes)
y_test = to_categorical(y_test, num_classes = num_classes)

model = Sequential()

model.add(Conv2D(
    32,
    kernel_size = 3,
    padding = "same",
    activation = "relu",
    input_shape = (32, 32, 3)
))
model.add(Conv2D(
    32,
    kernel_size = 3,
    activation = "relu"
))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(
    64,
    kernel_size = 3,
    padding = "same",
    activation = "relu"
))
model.add(Conv2D(
    64,
    kernel_size = 3,
    activation = "relu"
))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation("softmax"))

optimizer = Adam(lr = 0.001)
model.compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
)


weights_dir = './weights/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    period = 3
)


reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)


logging = TensorBoard(log_dir = "log/")

model.fit(
    X_train,
    y_train,
    verbose = 1,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = [early_stopping, reduce_lr, logging]
)
