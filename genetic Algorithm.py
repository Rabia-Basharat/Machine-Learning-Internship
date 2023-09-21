#!/usr/bin/env python
# coding: utf-8

# In[15]:


import random

bag_capacity = 30
num_items = 10
population_size = 40
num_generations = 50

items = [{'weight': random.randint(1, 10), 'value': random.randint(1, 20)} for i in range(num_items)]


population = [[random.choice([0, 1]) for i in range(num_items)] for i in range(population_size)]

def fitness(individual):
    total_weight = sum(items[i]['weight'] for i in range(num_items) if individual[i] == 1)
    total_value = sum(items[i]['value'] for i in range(num_items) if individual[i] == 1)
    
    if total_weight > bag_capacity:
        return 0
    return total_value


for generation in range(num_generations):
    
    fitness_scores = [fitness(individual) for individual in population]
    
    parents = random.choices(population, weights=fitness_scores, k=population_size // 2)
    
    new_population = []
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        crossover_point = random.randint(1, num_items - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        if random.random() < 0:  
            mutate_index = random.randint(0, num_items - 1)
            child1[mutate_index] = 1 - child1[mutate_index]
        
        if random.random() < 0:  
            mutate_index = random.randint(0, num_items - 1)
            child2[mutate_index] = 1 - child2[mutate_index]
        
        new_population.extend([child1, child2])
    
    population = new_population

best_individual = max(population, key=lambda ind: fitness(ind))
best_value = sum(items[i]['value'] for i in range(num_items) if best_individual[i] == 1)

# Print the result
print("Best individual:", best_individual)
print("Total weight of an item:", best_value)


# In[22]:


import numpy as np
import random  
rand_list = []  

for i in range(0,100):  
    Y_true = random.randint(1,100)  
    rand_list.append(Y_true)  
print(rand_list)  
rand_list = Y_true
 

MSE = np.square(np.subtract(Y_true,Y_pred)).mean()


# In[24]:


import numpy as np

num_samples = 100
X = np.random.rand(num_samples)  

print("Randomly generated X:" )
print(X)


# In[66]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
counter = 1

m = random.randint(-1, 1)
c = random.randint(1, 5)

def hypothesis(x):
    global m
    global c
    return m * x + c

def f(x):
    global counter
    counter+=1
    return x * 0.5

y_true = [f(x) for x in X] 
y_predicted = [hypothesis(x) for x in X]
y_predicted
plt.scatter(X, y_true)
plt.scatter(X, y_predicted)
plt.show()

MSE = np.square(np.subtract(Y_true,Y_pred)).mean()
print("MSE:", MSE)


# In[94]:


import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


num_samples = 100
X = np.random.rand(num_samples, 1)

m = random.randint(-1, 1)
c = random.randint(0, 5)

def hypothesis(x):
    global m
    global c
    return m * x + c

def f(x):
    return x * 0.5 + (-0.1, 0.1)

y_true = [f(x) for x in X]
y_predicted = [hypothesis(x) for x in X]

print(y_true, "\n")
print(y_predicted)


plt.scatter(X, y_true)
plt.scatter(X, y_predicted)

plt.title('True vs Predicted Values')
plt.show()


# In[71]:


import random
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("Housing.csv")
print(data)

m = random.randint(-1, 1)
c = random.randint(0, 5)

def hypothesis(x):
    global m
    global c
    return m * x + c

def f(x):
    return x * 0.5 + (-0.1, 0.1)

y_true = [f(x) for x in X]
y_predicted = [hypothesis(x) for x in X]

print(y_true, "\n")
print(y_predicted)


plt.scatter(X, y_true)
plt.scatter(X, y_predicted)

plt.title('True vs Predicted Values')
plt.show()


# In[73]:


import random
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


data = pd.read_csv("C:/Users/Rabia/Downloads/Housing.csv")
print(data)

# Assuming you have a column 'X' in your CSV file, extract it as a list
X = data['X'].tolist()

m = random.uniform(-1, 1)  # Use uniform for float values
c = random.uniform(0, 5)

def hypothesis(x):
    global m
    global c
    return m * x + c

def f(x):
    return 0.5 * x + random.uniform(-0.1, 0.1)  # Fix the f(x) function

y_true = [f(x) for x in X]
y_predicted = [hypothesis(x) for x in X]

print(y_true, "\n")
print(y_predicted)

plt.scatter(X, y_true, label='True Values')  # Label each scatter for legend
plt.scatter(X, y_predicted, label='Predicted Values')  # Label each scatter for legend

plt.title('True vs Predicted Values')
plt.legend()  # Show legend
plt.show()


# In[88]:


import random
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("C:\\Users\\Rabia\\Downloads\\Housing.csv")

print(data.columns)
print(data.head())


# X = data['area'].tolist()

# m = random.uniform(0, 1) 
# c = random.uniform(0, 5)
# m=1
# c=0

def hypothesis(x):
    global m
    global c
    return m * x + c

def f(x):
    return 0.5 * x 

y_true = [f(x) for x in X]
y_predicted = [hypothesis(x) for x in X]

#print(y_true, "\n")
#print(y_predicted)

plt.scatter(X, y_true, label='True Values') 
plt.scatter(X, y_predicted, label='Predicted Values') 

plt.title('True vs Predicted Values')
plt.legend() 
plt.show()


# In[93]:


from sklearn.datasets import make_regression
from matplotlib import pyplot
X_test, y_test = make_regression(n_samples=150, n_features=1, noise=0.2)
pyplot. scatter(X_test,y_test)
pyplot. show()


# In[115]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=None)
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

# plt.scatter(X_train, y_train, label='Training Data')
# plt.scatter(X_test, y_test, color='r', label='Testing Data')

# plt.title(' Regression Dataset') 
# plt.show()


# In[125]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class ManualLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weight =0
        self.bias = 0
        self.errors = []
        
        
    def fit(self, X, y):
#         n_samples, m = X.shape
#         self.weight = np.zeros(m)
        n_samples = len(X)
        for i in range(self.iterations):
            
            y_predicted = self.weight* X + self.bias
            self.errors.append(np.square(np.subtract(y,y_predicted)).mean())
            dw = (1/n_samples) * sum(X * (y_predicted - y))
            db = (1/n_samples) * sum(y_predicted - y)
            
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        print(self.errors)
            
            
    def predict(self, X):
        return self.weight * X + self.bias


X, y = make_regression(n_samples=50, n_features=1, noise=5, random_state=None)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

model = ManualLinearRegression(learning_rate=0.01, iterations=100)
model.fit(X_train, y_train)
model.weight

y_pred = model.predict(X_test)


plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, color='r', label='Testing Data')
plt.plot(X_test, y_pred, color='g', label='Linear Regression Line')
plt.show()


# In[155]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class ManualLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weight = 0
        self.bias = 0
        self.errors = []
        self.MSE = 0
        
    def fit(self, X, y):
        n_samples = len(X)
        for i in range(self.iterations):
            y_predicted = self.weight * X + self.bias
#             print('y_predicted')
#             print(y_predicted)
            self.MSE = self.errors.append(np.square(np.subtract(y, y_predicted)).mean())
            dw = (1/n_samples) * sum(X * (y_predicted - y))
            db = (1/n_samples) * sum(y_predicted - y)
#             print('db')
#             print(db)
            
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        print("MSE:", self.MSE)
            
    def predict(self, X):
        return self.weight * X + self.bias

X, y = make_regression(n_samples=50, n_features=1, noise=5, random_state=None)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

model = ManualLinearRegression(learning_rate=0.01, iterations=100)
model.fit(X_train, y_train)

y_pred = model.predict(X)
print('y_pred[0]')
print(y_pred)
print(X.shape)

plt.plot(model.errors, range(len(model.errors)))
plt.show()

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test, color='r')
plt.plot(X, y_pred.T[-10], color='g')
plt.show()



# In[ ]:




