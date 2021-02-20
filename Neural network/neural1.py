import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_img, train_labels), (test_img, test_labels) = data.load_data()

class_names = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boots']

train_img = train_img / 255.0
test_img = test_img / 255.0

plt.imshow(train_img[7], cmap=plt.cm.binary)
plt.show()