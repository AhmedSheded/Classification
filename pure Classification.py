import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('cfdata.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

data.head()
data.info()
data.describe()

positive = data[data['Admitted']==1]
negative = data[data['Admitted']==0]

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='g', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


def sigmoid(z):
    return (1 / 1+np.exp(-z))

z = np.arange(-5, 5, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.xlabel('z')
plt.ylabel('$\phi(z)$')
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()


def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


data.insert(0, 'Ones', 1)

X = data.drop('Admitted', axis=1)
y = data['Admitted']

X = np.array(X)
y = np.array(y)
theta = np.zeros(3)

thiscost = cost(theta, X, y)
print(thiscost)


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

costafteroptimize = cost(result[0], X, y)
print(costafteroptimize)


result = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg,
                      args=(X2, y2, learningRate))
print( 'result = ' , result )
print()

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))