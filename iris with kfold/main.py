import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

skf = StratifiedKFold(n_splits=5)

predict = np.zeros(y.shape[0])

for train, test in skf.split(X,y):
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    logreg = LogisticRegression().fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    predict[test] = y_pred
    print("train data \n", train)
    print("test data \n", test)
    print("result \n", y_pred)
    print("this accuracy \n", accuracy_score(y_test, y_pred))
    print("="*60)
    print()

print("total accuracy \n", accuracy_score(y, predict))

conf = confusion_matrix(y, predict)
print("confusion matrix is \n", conf)