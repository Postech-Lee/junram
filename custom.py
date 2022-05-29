import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np


print(tf.__version__)
print(keras.__version__)

image_datas = glob('C:\\Users\\user\\PycharmProjects\\datasets\\REAL\\*')
print(len(image_datas))
class_name = ["REAL", "FAKE"]
dic = {"REAL":0, "FAKE":1}

X=[]
Y=[]
for imagename in image_datas:
    image = Image.open(imagename)
    image = np.array(image)
    X.append(image)
    label = imagename.split('\\')[5]
    label = dic[label]
    Y.append(label)

image_datas = glob('C:\\Users\\user\\PycharmProjects\\datasets\\FAKE\\*')
print(len(image_datas))
class_name = ["REAL", "FAKE"]
dic = {"REAL":0, "FAKE":1}
try:
    for imagename in image_datas:
        image = Image.open(imagename)
        image = np.array(image)
        X.append(image)
        label = imagename.split('\\')[5]
        label = dic[label]
        Y.append(label)
except:
    pass

X=np.array(X)
Y=np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=44)

y_train = y_train[..., tf.newaxis]
y_test = y_test[..., tf.newaxis]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(x_train.shape, y_train.shape)
print( x_test.shape, y_test.shape)

x_mean = np.mean(x_train, axis=(0, 1, 2))
x_std = np.std(x_train, axis=(0, 1, 2))

x_train_full = (x_train-x_mean) / x_std
x_test = (x_test-x_mean) / x_std

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

print(x_test.shape)


def model_build():
    model = Sequential()

    input = Input(shape=(256, 256, 3))

    output = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(input)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(output)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')(output)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Flatten()(output)

    output = Dense(2048, activation='relu')(output)
    output = Dense(1024, activation='relu')(output)
    output = Dense(2, activation='softmax')(output)

    model = Model(inputs=[input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = model_build()
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=256,
                    validation_data=(x_test, y_test))


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'b--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r:', label='validation_accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.show()