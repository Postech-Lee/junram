
import tensorflow
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.datasets.fashion_mnist import load_data
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train[...,np.newaxis]
x_test = x_test[...,np.newaxis]

x_train = x_train / 255.
x_test = x_test / 255.

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def build_model():
    model = Sequential()
    input = Input(shape=(28, 28, 1))
    output = Conv2D(filters=32, kernel_size=(3, 3))(input)
    output = Conv2D(filters=64, kernel_size=(3, 3))(output)
    output = Conv2D(filters=64, kernel_size=(3, 3))(output)
    output = Flatten()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    model = Model(inputs=[input], outputs=output)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model

model_1 = build_model()
model_1.summary()

hist_1 = model_1.fit(x_train, y_train,
                     epochs=1,
                     validation_split=0.3,
                     batch_size=128)

hist_1.history.keys()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_1.history['loss'], 'b--', label='loss')
plt.plot(hist_1.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist_1.history['acc'], 'b--', label='loss')
plt.plot(hist_1.history['val_acc'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.show()