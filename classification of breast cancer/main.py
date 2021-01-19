from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, \
    precision_score, precision_recall_fscore_support,\
    precision_recall_curve, classification_report, zero_one_loss
import seaborn as sns
import matplotlib.pyplot as plt

# load breast cancer data

BreastData = load_breast_cancer()

X = BreastData.data
y = BreastData.target

# splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50, shuffle=True)

#applaying classification model

LogisticRegressionModel = LogisticRegression(penalty="l1", random_state=0, solver='liblinear').fit(X_train, y_train)


# calculatting details

print("Model Train Score is : ", LogisticRegressionModel.score(X_train, y_train))
print("Model Test Score is : ", LogisticRegressionModel.score(X_test, y_test))
print("Model Classes are : ", LogisticRegressionModel.classes_)
print('Model No of iteratios is : ', LogisticRegressionModel.n_iter_)

print("-"*60)

y_pred = LogisticRegressionModel.predict(X_test)
y_pred_prob = LogisticRegressionModel.predict_proba(X_test)
# print(y_pred[:10])
# print(y_pred_prob[:10])

# calculating confusion matrix

CM = confusion_matrix(y_test, y_pred)
print("Confusion matri is : \n", CM)

# drawing confusion matrix
sns.heatmap(CM, center=True)
plt.show()

# Calculatein accuracy Score : ((TP + TN)/ float(TP + TN + FP + FN))
AccSrore = accuracy_score(y_test, y_pred, normalize=False)
print("Accuracy Score is : ", AccSrore)

# Calculating f1_score : 2 * (precision * recall) / (precision + recall)

F1Score = f1_score(y_test, y_pred, average='micro')
print('F1 Score is : ', F1Score)

# Calculating recall score : (sensitivity) (TP / float (tp + FN)) 1/1+2

RecallScore = recall_score(y_test, y_pred, average='micro')
print("recall Score is : ", RecallScore)

# Calculating precision Score : (Specificity) (TP / float( TP + FP ))

PrecisionScore = precision_score(y_test, y_pred, average="micro")
print("Precision Score is : ", PrecisionScore)

# Calculate Precision recall Score :
PrecisionRecallScore = precision_recall_fscore_support(y_test, y_pred, average='micro')
print("Precision Recall Score is : ", PrecisionRecallScore)

# Calculate precision recall curve

PrecisionValue, RecallValue, ThresholdsValue = precision_recall_curve(y_test, y_pred)
print("precision Value is : ", PrecisionValue)
print("Recall Value is : ", RecallValue)
print("Thresholds Value is : ", ThresholdsValue)

# Calculate Classification Report :
Classification_report = classification_report(y_test, y_pred)
print("Classification Report is : ", Classification_report)

# Calculate zero one loss :

ZeroOneLossValue = zero_one_loss(y_test, y_pred, normalize=False)
print("zero One Loss Value : ", ZeroOneLossValue)