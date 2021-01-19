import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error,\
    mean_squared_error, median_absolute_error, accuracy_score
dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=4, shuffle=True)

# fitting logistic regression to the training set
clss = LogisticRegression(random_state=5, solver="newton-cg")
clss.fit(X_train, y_train)

# predicting
y_pred = clss.predict(X_test)
print(list(y_pred[:20]))
print(list(y_test[:20]))

print("N of iter is ",clss.n_iter_)
print("classes is ",clss.classes_)

cm = confusion_matrix(y_test, y_pred)
print("comfusion matrix is \n",cm)

print("mean absolute error is ", median_absolute_error(y_test, y_pred))
print("Accuracy score is ", accuracy_score(y_test, y_pred))