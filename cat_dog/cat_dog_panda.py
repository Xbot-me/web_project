from pickletools import optimize
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import np_utils

path = './animals/'
categories = ['dogs','panda','cats']

for category in categories:
    fig, _ = plt.subplots(3,4)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    test = os.listdir(path+category)[:1]
    #print(test)
    for k, v in enumerate(os.listdir(path+category)[:12]):
        print(k,v)
        img = plt.imread(path+category+'/'+v)
        plt.subplot(3,4,k+1)
        plt.axis('off')
        plt.imshow(img)
    #plt.show()

shape0 = []
shape1 = []

for category in categories:
    for files in os.listdir(path+category):
        shape0.append(plt.imread(path+category+'/'+files).shape[0])
        shape1.append(plt.imread(path+category+'/'+files).shape[1])
    print(category, '=> height min :',min(shape0), 'width min:',min(shape1))
    print(category, '=> height max :',max(shape0), 'width max:',max(shape1))
    shape0 = []
    shape1 = []

data = []
labels = []
imgPaths = []

HEIGHT =32
WIDTH = 55
N_CHANNELS = 3

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imgPaths.append([path+category+'/'+f,k])

import random
random.shuffle(imgPaths)
print(imgPaths[:10],'\n')

for imgPath in imgPaths:
    img = cv2.imread(imgPath[0])
    img = cv2.resize(img,(WIDTH,HEIGHT))
    data.append(img)
    label = imgPath[1]
    labels.append(label)

data = np.array(data, dtype='float')/255.0
labels = np.array(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.2,random_state=42)

trainY=np_utils.to_categorical(trainY,3)
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

model = Sequential()

model.add(Conv2D(
    32,
    (2,2),
    activation='relu',
    input_shape=(HEIGHT,WIDTH,N_CHANNELS)
))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(trainX,trainY,batch_size=100,epochs=100,verbose=1)

from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score
pred = model.predict(testX)
predictions = argmax(pred,axis=1)
cm = confusion_matrix(testY,predictions)
fig = plt.figure()
fig.patch.set_facecolor('xkcd:white')
ax= fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Model confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels(['']+categories)
ax.set_yticklabels(['']+categories)

for i in range(3):
    for j in range(3):
        ax.text(i,j,cm[i,j],va='center',ha='center')

plt.xlabel('Predicted')
plt.ylabel('True')
#plt.show()

accuracy = accuracy_score(testY,predictions)
print("Accuracy : %.2f%%"%(accuracy*100.0))
model.save('model_animal_cnn_1.h5')
