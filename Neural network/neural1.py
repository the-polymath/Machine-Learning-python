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

'''
plt.imshow(train_img[7], cmap=plt.cm.binary)
plt.show()
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

model.fit(train_img, train_labels, epochs=8)

test_loss, test_acc = model.evaluate(test_img, test_labels)
print(test_loss, test_acc)

model_predict = model.predict(test_img)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_img[i], cmap=plt.cm.binary)
    plt.xlabel("actual" + class_names[i])
    plt.title("Prediction" + class_names[np.argmax(model_predict[i])])
    plt.show()
