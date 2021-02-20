import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



class clust():
    def _load_data(self, datas):
        predict = "Calories"
        X = np.array(datas.drop([predict], 1))
        y = np.array(datas[predict])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,
                                                                                random_state=42)

    def __init__(self, datas):
        self._load_data(datas)

    def classify(self, model = RandomForestClassifier(n_estimators=100)):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))

    def Kmeans(self, output='add'):
        n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters=3, random_state=42)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)

        if output == 'replace':
            self.X_train = y_labels_train[:, np.newaxis]
            self.X_test = y_labels_test[:, np.newaxis]
        else:
            raise ValueError('output should be either add or replace')
        return self


data = pd.read_csv("1000_dataset.csv")
data.info()
lst = list(data.columns)
print(lst)
features = lst[3:9]
print(features)
data1 = data[features]
print(data1)
clust(data1).Kmeans(output='replace').classify()
