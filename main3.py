from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

dataset = load_iris()
batch_size = 2

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=1)

clf = MLPClassifier(random_state=1, max_iter=1000, batch_size=batch_size).fit(X_train, y_train)
pred_mlp = clf.predict(X_test)

print(y_test)
print(pred_mlp)

print(confusion_matrix(y_test, pred_mlp))
print(classification_report(y_test, pred_mlp))