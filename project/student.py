import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())
data1 = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data1.drop([predict], 1))
y = np.array(data1[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''
best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    lm = linear_model.LinearRegression()

    lm.fit(x_train, y_train)
    acc = lm.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(lm, f) '''

pickle_in = open("studentmodel.pickle", "rb")
lm = pickle.load(pickle_in)

acc = lm.score(x_test, y_test)
print(acc)

print("coefficient : ", lm.coef_)
print("Intercept : ", lm.intercept_)

pred = lm.predict(x_test)

for i in range(len(pred)):
    print(pred[i], x_test[i], y_test[i])

style.use("ggplot")
pyplot.scatter(data1["G2"], data1[predict])
pyplot.xlabel("G2")
pyplot.ylabel("Final Grade")
pyplot.show()
