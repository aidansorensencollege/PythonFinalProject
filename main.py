# CS2021 Final Project
__author__ = "Jordan Myers and Aidan Sorensen"
__copyright__ = "Copyright 2019, The University of Cincinnati"
__email__ = "myers3jr@mail.uc.edu, sorensaa@mail.uc.edu"

# The libraries used in this python project are tensorflow,
# matplotlib, numpy, and PIL.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from numpy import asarray
from PIL import Image


# Here we are importing the MNIST database of handwritten digits.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Here we have created a model-related API using the Sequential
# class built into keras. This neural network uses a linear stack of
# four layers, one layer to flatten the input images into a layer of
# 784 neurons and three layers of size 512, 256, and 128 respectively.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),

  # These three layers are normal densely-connected NN layers.
  # The activation function uses reLU, which is a Rectified Linear Unit.
  # It acts a linear function to decide whether or not the values are good
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),

  # We use softmax to find the largest correlation from the other layers
  tf.keras.layers.Dense(10, activation='softmax')
])


# This compile is going to take the model and apply the adam algorithm to 
# the dataset and compute differentials and max and mins of the correlation graph.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Here, we train the NN using the model.fit method and 5 ephocs,
# and then evaluate the test results using model.evaluate.
model.fit(x_train, y_train, epochs = 5)
model.evaluate(x_test, y_test, verbose = 2)


# Opens the test image(s) from the directory, converts to greyscale, and converts to array.
img_name = input('\nEnter the image filename: ')
test_img = asarray(np.invert(Image.open(img_name).convert('L')))

# This line uses the matplot library to show the image that is opened/run through the NN.
plt.imshow(test_img, cmap='Greys')

# Because keras models are optimized to make predictions on a batch, we put the image
# into a list of one element on this line.
test_img = np.expand_dims(test_img, 0)

# Here, the image is run through the NN using model.predict and the prediction
# is printed out using np.argmax on the array output from the final layer of the NN.
predictions_single = model.predict(test_img)
print('Prediction for test image:', np.argmax(predictions_single[0]))

# Finally, we run the plt module to allow us to see the image that was
# analyzed by the neural network.
plt.show()
