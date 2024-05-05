# This code is based on
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import numpy as np
from scipy.ndimage import zoom
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, get_custom_objects
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.initializers import Initializer

nb_classes = 72
# input image dimensions
img_rows, img_cols = 32, 32

ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * 160):
    X_train[i] = zoom(ary[i], (img_rows / ary[i].shape[0], img_cols / ary[i].shape[1]))

Y_train = np.repeat(np.arange(nb_classes), 160)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

def my_init(shape, dtype=None):
    return np.random.normal(loc=0.0, scale=0.1, size=shape).astype(np.float32)

class MyInitializer(Initializer):
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, shape, dtype=None):
        return np.random.normal(loc=0.0, scale=self.scale, size=shape).astype(np.float32)

    def get_config(self):
        return {"scale": self.scale}


def m6_1():
    model.add(Conv2D(32, (3, 3), kernel_initializer=my_init, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), kernel_initializer=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), kernel_initializer=my_init))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), kernel_initializer=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=my_init))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

# with custom_object_scope({'my_init': my_init}):
#     model = load_model('hiraga_ETL8G_400.h5')

# my_initをKerasの初期化関数として登録する
get_custom_objects().update({"my_init": MyInitializer()})
model = load_model('hiraga_ETL8G_400.h5')

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(datagen.flow(X_train, Y_train, batch_size=16), steps_per_epoch=X_train.shape[0] // 16,
          epochs=400, validation_data=(X_test, Y_test))
model.save('hiraga_ETL8G_add.h5')