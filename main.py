import MBGD
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import unique
from sklearn.utils import shuffle
import numpy as np

dataset = load_iris()

X = dataset.data
y = dataset.target

new_X, new_y = shuffle(X, y, random_state=1)

splitted_X = np.array_split(new_X, 10)
splitted_y = np.array_split(new_y, 10)

# X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=1)

# print(X_test)
# print(y_test)

# print(splitted_X[0])
# print(splitted_y[0])

for i in range(10):
  X_train = splitted_X[i]
  y_train = splitted_y[i]

  X_test = []
  y_test = []
  for j in range(10):
    if (i != j):
      for item in splitted_X[i]:
        X_test.append(item.tolist())  
      y_test = np.concatenate((y_test, splitted_y[i]), axis=None) 

  mbgd = MBGD.MBGD()
  mbgd.setBias(1)
  totalDiffTargetClass = len(unique(y_train))
  hidden1 = mbgd.createHiddenLayer(16, 2)
  output = mbgd.createOutputLayer(4, totalDiffTargetClass)
  mbgd.setLayer([hidden1, output])

  mbgd.fit(X_train, y_train)
  mbgd.printmodel()
  mbgd_pred = mbgd.predict(X_test) 
  c_matrix = mbgd.confusion_matrix(y_test, mbgd_pred)

  print(mbgd_pred)
  print(c_matrix)
  print(mbgd.classification_report(c_matrix))

  filename = "model" + str(i) + ".json"
  print(mbgd.saveModel(filename))