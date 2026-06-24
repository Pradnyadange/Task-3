import numpy as np
import torch
print("Neural Network Symmetry Detection")
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden  = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def sigmoid_derivative(self, y):
        return y * (1.0 - y)
    def feedforward(self, X):
        self.hidden_input  = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input   = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output  = self.sigmoid(self.final_input)
        return self.final_output
    def backward(self, X, y, learning_rate):
        output_error = self.final_output - y                               
        output_delta = output_error * self.sigmoid_derivative(self.final_output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output -= np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output           -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden  -= np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden           -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backward(X, y, learning_rate)
    @staticmethod                                                           
    def Symmetry():
        print("Training Neural Network to Detect Symmetry in 6-bit Binary Numbers...")
        print("Target sweeps: 1425")
        X = torch.tensor(                                                   
            [[int(b) for b in format(i, '06b')] for i in range(64)],
            dtype=torch.float32
        )
        Y = torch.tensor(
            [1.0 if list(x.numpy()) == list(x.numpy())[::-1] else 0.0 for x in X],
            dtype=torch.float32
        ).unsqueeze(1)
        net = NeuralNetwork(6, 2, 1)
        net.train(X.numpy(), Y.numpy(), epochs=1425, learning_rate=0.9)
        predictions = net.feedforward(X.numpy())
        acc = np.mean((predictions > 0.5) == Y.numpy()) * 100
        print(f"Final Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    NeuralNetwork.Symmetry()                                                