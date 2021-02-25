import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=100000)

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)
# print(len(test_data[0]), len(test_data[1]))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "??") for i in text])

'''
# neural model
model = keras.Sequential()
model.add(keras.layers.Embedding(100000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

print(len(x_train), len(x_val))

training = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("neural_model.h5")
'''

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


model = keras.models.load_model("neural_model.h5")

with open("test_review.txt", encoding="utf-8") as file:
    for line in file:
        pl = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("\"", '').replace(":", "").strip()
        en = review_encode(pl)
        en = keras.preprocessing.sequence.pad_sequences([en], value=word_index["<PAD>"], padding="post", maxlen=256)
        predict = model.predict(en)
        print(line)
        print(en)
        print(predict)

'''
for i in range(5):
    test_review = test_data[i]
    predict = model.predict(test_review)
    print("review : ", decode_review(test_review))
    print("prediction : " + str(predict[i]))
    print("actually : " + str(test_labels[i]))
    print(results)
'''

