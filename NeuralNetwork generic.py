#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[31]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Sigmoid:
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, output_delta):
        return output_delta * self.output * (1 - self.output)

class Softmax:
    def forward(self, input_data):
        if input_data.ndim == 1:
            # If input_data is 1D, treat it as a single sample
            exp_x = np.exp(input_data - np.max(input_data))
            self.output = exp_x / exp_x.sum()
        else:
            # If input_data is 2D, treat each row as a sample
            exp_x = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
            self.output = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self.output

    def backward(self, output_delta):
        batch_size = output_delta.shape[0]
        return (self.output - output_delta) / batch_size


class NeuralLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((output_size,))
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

    nn.add_layer(NeuralLayer(X_train.shape[1], 4, Sigmoid()))
    nn.add_layer(NeuralLayer(4, 6, Softmax()))
    nn.add_layer(NeuralLayer(6, 7, Sigmoid()))
    nn.add_layer(NeuralLayer(7, 8, Sigmoid()))
    nn.add_layer(NeuralLayer(8, 8, Sigmoid()))
    nn.add_layer(NeuralLayer(8, num_classes, Softmax()))

    num_iterations = 1000
    learning_rate = 0.005
    batch_size = 8

    nn.train(X_train, y_train, num_iterations, learning_rate, batch_size)

    correct_predictions = 0
    total_samples = X_test.shape[0]

    for i in range(total_samples):
        test_input = X_test[i]
        true_label = y_test[i]

        predicted_output = nn.predict(test_input)
        predicted_label = np.argmax(predicted_output)
        

        if predicted_label == np.argmax(true_label):
            correct_predictions += 1
    print(predicted_output)
    print(predicted_label)
    accuracy = correct_predictions / total_samples
    print(f"Accuracy on the test set: {accuracy:.2f}")


# In[ ]:




