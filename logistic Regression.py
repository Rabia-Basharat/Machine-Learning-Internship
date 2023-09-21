#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load breast cancer dataset
# bc = datasets.load_breast_cancer()
bc = datasets.load_iris()
bc_df = pd.DataFrame(data=bc.data, columns=bc.feature_names)
bc_df['target'] = bc.target
print('bc data:', bc_df)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            linear_prediction = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_prediction)
            
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cross entropy loss
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            self.losses.append(loss)

    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_prediction)
        class_pred = [1 if y > 0.5 else 0 for y in y_pred]
        return class_pred

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
print('train X:' , X_train)


clf = LogisticRegression(learning_rate=0.001, iterations=1000)
clf.fit(X_train, y_train)


# plt.scatter(X_train.iloc[:, 0], y_train, color='blue', label='Train Data')
# plt.plot(X_train.iloc[:, 0], clf.predict(X_train), Scolor='red', linewidth=2, label='Predictions')
# plt.xlabel('Feature 0')
# plt.ylabel('Target')
# plt.show()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns 
# Example dataset
data = sns.load_dataset("iris") 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            linear_prediction = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_prediction)
            
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cross entropy loss
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            self.losses.append(loss)

    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_prediction)
        class_pred = [1 if y > 0.5 else 0 for y in y_pred]
        return class_pred

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
print('train X:' , X_train)


clf = LogisticRegression(learning_rate=0.001, iterations=1000)
clf.fit(X_train, y_train)


# Scatter plot with hue 
sns.scatterplot(x="sepal_length", y="sepal_width", data=data, hue="species")
# plt.plot(X_test, color = 'red', linewidth = 2, label = 'prediction')
plt.title("Scatter Plot with Hue")
plt.show()


# In[ ]:




