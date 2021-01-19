from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)

clf = LogisticRegression(random_state=1, solver="saga").fit(X, y)
print(clf.predict(X[:10, :]))
print(clf.predict_proba(X[:10, :]))
score = clf.score(X, y)
print('Score = ', score)
print("No of iterations = ", clf.n_iter_)
print("Classes = ", clf.classes_)