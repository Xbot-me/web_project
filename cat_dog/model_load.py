import numpy as np
import tensorflow as tf

from tensorflow import keras

model = keras.models.load_model('./model_animal_cnn_1.h5')
import cv2
test_data = []
test_img_o = cv2.imread('./panda.jpeg')
HEIGHT =32
WIDTH = 55
N_CHANNELS = 3
test_img = cv2.resize(test_img_o,(WIDTH,HEIGHT))

test_data.append(test_img)

test_data = np.array(test_img,dtype="float")/255.0
test_data = test_data.reshape([-1,32,55,3])

pred = model.predict(test_data)
from numpy import argmax
predictions = argmax(pred,axis=1)
path = './animals/'
categories = ['dogs','panda','cats']

print('Prediction :'+categories[predictions[0]])

animals = ['dog','panda','cat']

for idx,animal, x in zip(range(0,3),animals, pred[0]):
    print("ID: {}, Label: {} -> {}%".format(idx,animal,round(x*100,2)))
print('Prediction : '+animals[predictions[0]])