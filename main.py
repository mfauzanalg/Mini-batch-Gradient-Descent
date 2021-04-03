import MBGD
import pandas as pd

# df = pd.read_csv("./dataset.csv")
# label = df["label"]
# data = df.drop(columns=["label"], inplace=False)
# label = label.values
# data = data.values

# mbgd = MBGD.MBGD()
# mbgd.setBias(1)

# hidden1 = mbgd.createHiddenLayer(2, 3)
# output = mbgd.createOutputLayer(2)

# mbgd.setLayer([hidden1, output])

# mbgd.fit(data, label)

# mbgd.printmodel()
# print(mbgd.predict(data))

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import unique

dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=1)

mbgd = MBGD.MBGD()
mbgd.setBias(1)

totalDiffTargetClass = len(unique(y_train))

hidden1 = mbgd.createHiddenLayer(2, 2)
output = mbgd.createOutputLayer(4, totalDiffTargetClass)

mbgd.setLayer([hidden1, output])

mbgd.fit(X_train, y_train)

mbgd.printmodel()
mbgd_pred = mbgd.predict(X_test) 
c_matrix = mbgd.confusion_matrix(y_test, mbgd_pred)

print(mbgd_pred)
print(c_matrix)
print(mbgd.classification_report(c_matrix))
print(mbgd.saveModel("savedmodel.json"))