from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0
num_classes = 10
epochs = 100
batch_size = 32
class LeNet5(Model):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = Conv2D(6, kernel_size=(5,5), padding='same', activation= 'relu')
        self.conv2 = Conv2D(16, kernel_size=(5,5), activation='relu')
        self.max_pool = MaxPooling2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(84, activation='relu')
        self.dense3 = Dense(num_classes, activation='softmax')

    def call(self, input_data):
        x = self.max_pool(self.conv1(input_data))
        x = self.max_pool(self.conv2(x))
        x = self.flatten(x)
        x = self.dense3(self.dense2(self.dense1(x)))
        return x

model = LeNet5(num_classes)
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir='.logs', histogram_freq=1)]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_t
                           est),
          callbacks=callbacks)

