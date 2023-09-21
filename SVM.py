#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[149]:


iris = datasets.load_iris()
X = iris.data
y = iris.target
# iris_df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width'])
# iris_df['target'] = iris.target
# condition = iris_df['target'] == 2
# iris_df = iris_df[~condition]
# iris_df.head()
iris_df['target'].unique()
y


# In[96]:


# binary_df = iris_df[iris_df['target'] != 2]
# print(binary_df.head())
# binary_df['target'].unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[150]:


class SVM:
    def __init__(self, learning_rate=0.01, lambda_parameter = 0.00005, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_parameter = lambda_parameter
        self.weights = None
        self.bias = None
        self.losses = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y<= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.iterations):
            for idx, x_i in enumerate (X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_parameter * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_parameter * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]
           

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)
        


# In[151]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


# In[152]:


model = SVM()
model.fit(X_train, y_train)
prediction = model.predict(X_test)


# In[153]:


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)
print("SVM accuracy:", accuracy(y_test, prediction))


# In[154]:


def visualize_SVM():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1],  marker = '+', c=y)
    
    
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])
    
    x1_1 = get_hyperplane_value(x0_1, model.weights, model.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, model.weights, model.bias, 0)
    
    x1_1_m = get_hyperplane_value(x0_1, model.weights, model.bias, -1)
    x1_2_m = get_hyperplane_value(x0_2, model.weights, model.bias, -1)
    
    x1_1_p = get_hyperplane_value(x0_1, model.weights, model.bias, 1)
    x1_2_p = get_hyperplane_value(x0_2, model.weights, model.bias, 1)
    
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "b")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "r")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "r")
    
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])
    

    plt.show()
    
visualize_SVM()    
    
    
    
    


# In[105]:


import seaborn as sns
sns.scatterplot(data=iris_df, x='sepal_length', y='sepal_width', hue='target')
plt.title("Iris Dataset")
plt.show()


# In[ ]:


1. If in doubt, Code
2. Experiment
3. Visualize

