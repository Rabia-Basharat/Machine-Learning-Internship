#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

class Sigmoid:
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, output_delta):
        return output_delta * self.output * (1 - self.output)

    
class Softmax:
    def forward(self, input_data):
        self.input = input_data
        if input_data.ndim == 1:
            # If input_data is 1D then treat it as a single sample
            exp_x = np.exp(input_data - np.max(input_data))
            self.output = exp_x / exp_x.sum()
        else:
            # If input_data is 2D treat each row as a sample
            exp_x = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
            self.output = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self.output

    def backward(self, output_delta):
        batch_size = output_delta.shape[0]
        return (self.output - output_delta) / batch_size


class NNLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.biases = np.random.uniform(-1, 1, (output_size,))
        self.activation_function = activation_function

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.activation_function.forward(self.output)

    def backward(self, output_delta):
        delta = output_delta * self.activation_function.backward(self.output)
        input_delta = np.dot(delta, self.weights.T)
        weight_gradients = np.dot(self.input.T, delta)
        bias_gradients = delta.sum(axis=0)
        return input_delta, weight_gradients, bias_gradients


class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, input_data, target_output, num_iterations, learning_rate, batch_size):
        num_samples = len(input_data)
        
        for iteration in range(num_iterations):
            for batch_start in range(0, num_samples, batch_size):
                batch_end = batch_start + batch_size
                batch_input = input_data[batch_start:batch_end]
                batch_target = target_output[batch_start:batch_end]

                layer_input = batch_input
                for layer in self.layers:
                    layer_input = layer.forward(layer_input)

                predicted_output = layer_input
                loss = -np.sum(batch_target * np.log(predicted_output + 1e-10))
                output_delta = predicted_output - batch_target

                for layer in reversed(self.layers):
                    output_delta, weight_gradients, bias_gradients = layer.backward(output_delta)
        
                    layer.weights -= learning_rate * weight_gradients
                    layer.biases -= learning_rate * bias_gradients

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss = {loss:.4f}")

    def predict(self, input_data):
        if len(input_data) != self.layers[0].weights.shape[0]:
            raise ValueError("Input data size does not match the input layer size.")

        layer_input = input_data
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
        return layer_input


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    num_classes = len(np.unique(y))
    y_one_hot = np.eye(num_classes)[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    nn = NeuralNetwork()

    input_size = X_train.shape[1] 
    hidden_size = 4 
    output_size = num_classes 

    activation_function_hidden = Sigmoid()
    activation_function_output = Softmax()

    nn.add_layer(NNLayer(input_size, hidden_size, activation_function_hidden))
    nn.add_layer(NNLayer(hidden_size, output_size, activation_function_output))
    

    num_iterations = 100
    learning_rate = 0.01
    batch_size = 32

    nn.train(X_train, y_train, num_iterations, learning_rate, batch_size)

    correct_predictions = 0
    total_samples = X_test.shape[0]
    
    test_input = np.array([0.8, 0.2, 0.4, 0.6]) 
    result = nn.predict(test_input)
    print("Predicted Output:")
    print(result)

    for i in range(total_samples):
        test_input = X_test[i]
        true_label = y_test[i]

        predicted_output = nn.predict(test_input)
        predicted_label = np.argmax(predicted_output)

        if predicted_label == np.argmax(true_label):
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    print(f"Accuracy on the test set: {accuracy:.2f}")


# In[13]:


import pickle 

with open('C:\\Users\\Rabia\\Downloads\\Internship\\NNtrained_model.pkl', 'wb') as model_file:
        pickle.dump(nn, model_file)


# In[8]:


with open('NNtrained_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)


# In[17]:


from sklearn.metrics import precision_score, recall_score, f1_score

true_labels = np.argmax(y_test, axis=1)
predicted_labels = []

for i in range(total_samples):
    test_input = X_test[i]
    predicted_output = nn.predict(test_input)
    predicted_label = np.argmax(predicted_output)
    predicted_labels.append(predicted_label)

precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# In[16]:


from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion)


# In[ ]:




