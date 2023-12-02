import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

(X_train, Y_train), (X_test, Y_test) =  mnist.load_data()

type(X_train)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

"""Training data = 60,000 Images

Test data = 10,000 Images

Image dimension  --> 28 x 28

Grayscale Image  --> 1 channel
"""

print(X_train[10])

print(X_train[10].shape)

plt.imshow(X_train[25])
plt.show()


print(Y_train[25])

"""Image Lables"""

print(Y_train.shape, Y_test.shape)

print(np.unique(Y_train))

print(np.unique(Y_test))

"""We can use these labels as such or we can also apply One Hot Encoding

All the images have the same dimensions in this dataset, If not, we have to resize all the images to a common dimension
"""

X_train = X_train/255
X_test = X_test/255

print(X_train[10])

import matplotlib.pyplot as plt


sample_indices = [0, 1, 2,10, 4]

num_samples = len(sample_indices)
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

for i, index in enumerate(sample_indices):
    axes[i].imshow(X_train[index], cmap='gray')
    axes[i].set_title(f"Label: {Y_train[index]}")
    axes[i].axis('off')

plt.show()

"""Building the Neural Network"""

# setting up the layers of the Neural  Network

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)

"""Training data accuracy = 98.9%

**Accuracy on Test data:**
"""

loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)

"""Test data accuracy = 97.1%"""

print(X_test.shape)

plt.imshow(X_test[0])
plt.show()

print(Y_test[0])

Y_pred = model.predict(X_test)

print(Y_pred.shape)

print(Y_pred[0])

"""model.predict() gives the prediction probability of each class for that data point"""

label_for_first_test_image = np.argmax(Y_pred[0])
print(label_for_first_test_image)

Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

conf_mat = confusion_matrix(Y_test, Y_pred_labels)

print(conf_mat)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

import matplotlib.pyplot as plt
import numpy as np

class_counts = np.bincount(Y_train)

plt.figure(figsize=(10, 6))
plt.bar(range(10), class_counts, tick_label=range(10))
plt.xlabel('Digit Class')
plt.ylabel('Number of Samples')
plt.title('Data Distribution in MNIST Training Dataset')
plt.show()

"""Building a Predictive System"""

import matplotlib.pyplot as plt

history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

import matplotlib.pyplot as plt

history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', marker='o', linestyle='-')
plt.plot(val_loss, label='Validation Loss', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Visualization')
plt.legend()
plt.grid(True)
plt.show()

input_image_path = '/content/Screenshot 2023-07-14 181442.png'

input_image = cv2.imread(input_image_path)

type(input_image)

print(input_image)

cv2_imshow(input_image)

input_image.shape

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

grayscale.shape

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize.shape

cv2_imshow(input_image_resize)

input_image_resize = input_image_resize/255

type(input_image_resize)

image_reshaped = np.reshape(input_image_resize, [1,28,28])

input_prediction = model.predict(image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize = input_image_resize/255

image_reshaped = np.reshape(input_image_resize, [1,28,28])

input_prediction = model.predict(image_reshaped)

input_pred_label = np.argmax(input_prediction)

print('The Handwritten Digit is recognised as ', input_pred_label)
