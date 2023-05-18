
from gc import callbacks
from tabnanny import verbose
import tensorflow as tf
#print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train= x_train/255
x_test = x_test/255
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y_train=lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(x_train)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2,verbose=1,factor=0.5,min_lr=0.0001)

model = Sequential()
model.add(Conv2D(
    32,
    kernel_size=(3,3),
    kernel_initializer = 'he_normal',
    input_shape=(28,28,1),
    activation='relu',
    strides= 1,
    padding='same'
))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,
    kernel_size=(3,3),
    strides=1,
    padding='same',
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(.25))

model.add(Conv2D(128,
    kernel_size=(3,3),
    strides=1,
    padding='same',
    activation = 'relu'
))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.4))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

print(x_test.shape)
print(y_test.shape)
history = model.fit(x_train,y_train,batch_size=128,epochs=200,validation_data=(x_test,y_test),callbacks=[learning_rate_reduction],verbose=1)
model.save('./mymodel')
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")