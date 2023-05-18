import numpy as np
import tensorflow as tf

from tensorflow import keras

model = keras.models.load_model('./')

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()


x_test = x_test/255

x_test = x_test.reshape(-1,28,28,1)

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

y_test = lb.fit_transform(y_test)

predictions = model.predict(x_test)
predictions=np.argmax(predictions,axis=1)
predictions +=1

print(predictions.shape,x_test.shape)

import matplotlib.pyplot as plt


plt.imshow(x_test[1].reshape(28,28),cmap='gray')
plt.title(np.where(y_test[1] == 1)[0][0])
plt.axis('off')
plt.show()

