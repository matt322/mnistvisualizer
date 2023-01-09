import numpy as np

class Layer: # base fc layer class
    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weights = np.random.rand(inputSize, outputSize)/2 - 0.25
        self.biases = np.random.rand(1, outputSize)/2 - 0.25

    def forwardProp(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backProp(self, error, learning_rate):
        dEdWeights = np.dot(self.input.T, error)
        dEdInput = np.dot(error, self.weights.T)
        self.weights -= learning_rate * dEdWeights
        self.biases -= learning_rate * error # derivative of biases is just the error
        return dEdInput

class ReLU(Layer):
    def __init__(self):
        pass
    def forwardProp(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output
    
    def backProp(self, error, learning_rate):
        return (self.input > 0) * error

class Softmax(Layer):
    def __init__(self):
        pass
   
    def forwardProp(self, input):
        self.input = input
        input = input-np.max(input)
        self.output = np.exp(input) / np.sum(np.exp(input))
        return self.output

    def backProp(self, error, learning_rate):
        return error
    
    