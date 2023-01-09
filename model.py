from layer import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Model:

    def __init__(self, learningRate = 0.005, inputSize = 784) -> None:
        self.layers = []
        self.learningRate = learningRate
        self.inputSize = inputSize
    
    def addLayer(self, layer:Layer):
        self.layers.append(layer)

    def cross_entropy(self, true, pred):
        return -np.sum(true * np.log(pred))
    
    def crossentDeriv(self, true, pred):
        return pred-true

    def train(self, xTrain, yTrain, xTest, yTest, epochs, epochSize, history):
        for epoch in range(epochs):
            loss = 0
            for _ in range(epochSize):
                i = np.random.randint(0, len(xTrain))
                output = self.predict(xTrain[i])  
                loss += self.cross_entropy(yTrain[i], output)
                error = self.crossentDeriv(yTrain[i], output)
                for layer in reversed(self.layers):
                    error = layer.backProp(error, self.learningRate * 0.99**(epoch))
            acc = 0
            for j in range(len(xTest)):
                r = self.predict(xTest[j])
                if np.argmax(r) == np.argmax(yTest[j]):
                    acc += 1
            loss /= epochSize
            acc /= len(xTest)
            history.append(acc)
            print('epoch %d/%d   loss=%f   acc=%f   lr=%f' % (epoch+1, epochs, loss, acc, self.learningRate * 0.99**(epoch)))

    def predict(self, input):
        res = input
        for layer in self.layers:
            res = layer.forwardProp(res)
        return res

if __name__ == "__main__":
    from keras.datasets import mnist
    from keras.utils import np_utils
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    model = Model()
    model.addLayer(Layer(784, 512))
    model.addLayer(ReLU())
    model.addLayer(Layer(512, 128))
    model.addLayer(ReLU())
    model.addLayer(Layer(128, 128))
    model.addLayer(ReLU())
    model.addLayer(Layer(128, 128))
    model.addLayer(ReLU())
    model.addLayer(Layer(128, 64))
    model.addLayer(ReLU())
    model.addLayer(Layer(64, 10))
    model.addLayer(Softmax())
    history = []
    model.train(x_train, y_train, x_test, y_test, 400, 1000, history)

    file = open('512x128x128x128x64', 'wb')
    pickle.dump(model, file)
    file.close()

    plt.plot(history)
    plt.show()