from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import math

batch_size=531
class CustomDataloader(Sequence):
	def __init__(self, x_set, y_set, batch_size, shuffle=False):
	    self.x, self.y = x_set, y_set
	    self.batch_size = batch_size
	    self.shuffle = shuffle


    def __len__(self):
       return math.ceil(len(self.x) / self.batch_size)


    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]
        return np.array(batch_x), np.array(batch_y)


def model_build():
    model = Sequential()

    input = Input(shape=(32, 32, 3))

    output = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(output)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(output)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Flatten()(output)

    output = Dense(256, activation='relu')(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    model = Model(inputs=[input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model