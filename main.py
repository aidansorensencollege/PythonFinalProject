# CS2021 Final Project
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

sess = tf.compat.v1.Session()

import numpy as np
from numpy import asarray
from PIL import Image

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  #extra layer, each layer adds considerable time but makes the learning more accurate
  #this activation is relu instead of sigmoid because the function is more accurate and more efficient
  tf.keras.layers.Dense(10, activation='softmax')#use softmax to find the largest correlation from the other layers
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

# Opens the test image(s) from the directory, converts to greyscale, and converts to array.
img_name = input('\nEnter the image filename: ')
test_img = asarray(np.invert(Image.open(img_name).convert('L')))

# This line uses the matplot library to show the image that is opened/run through the NN
plt.imshow(test_img, cmap='Greys')

# Because keras models are optimized to make predictions on a batch, we put the image
# into a list of one element on this line
test_img = np.expand_dims(test_img, 0)

# Here, the image is run through the NN using model.predict and the prediction
# is printed out using np.argmax on the array output from the final layer of the NN
predictions_single = model.predict(test_img)
print('Prediction for test image:', np.argmax(predictions_single[0]))

plt.show()
