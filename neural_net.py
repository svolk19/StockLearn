import numpy as np
import matplotlib.pyplot as plt
import utils.activations as act
import utils.gradient_descents as sgd


class NeuralNetwork(object):
    def __init__(self, input_size, output_size, hidden1_size, hidden2_size):

        # layer sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # initialize random primary weight and bias scheme
        w1 = np.random.randn(self.input_size, self.hidden1_size)
        w2 = np.random.randn(self.hidden1_size, self.hidden2_size)
        w3 = np.random.randn(self.hidden2_size, self.output_size)

        self.weights = np.array([w1, w2, w3])

        b1 = np.zeros(self.hidden1_size)
        b2 = np.zeros(self.hidden2_size)
        b3 = np.zeros(self.output_size)

        self.biases = np.array([b1, b2, b3])

        # initialize forward propagation parameters to 0.0
        self.z2 = 0.0
        self.a2 = 0.0
        self.z3 = 0.0
        self.a3 = 0.0
        self.z4 = 0.0
        self.a4 = 0.0

    def predict(self, X):
        # feed forward data set X
        # assumes X is a numpy array, returns numpy array

        self.z2 = np.dot(X, self.weights[0]) + self.biases[0]
        self.a2 = act.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights[1]) + self.biases[1]
        self.a3 = act.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.weights[2]) + self.biases[2]
        self.a4 = act.sigmoid(self.z4)
        return self.a4

    def back_propagate(self, X, y):

        # back propagation of value
        self.yhat = self.predict(X)

        delta4 = np.multiply(-(y - self.yhat),
                             act.sigmoid(self.z4, deriv=True))
        dJdW3 = np.dot(self.a3.T, delta4)
        dJdB3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = np.dot(delta4, self.weights[2].T) * act.sigmoid(self.z3, deriv=True)
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdB2 = np.sum(delta3, axis=0)
        delta2 = np.dot(delta3, self.weights[1].T) * act.sigmoid(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)
        dJdB1 = np.sum(delta2, axis=0)

        weight_gradients = np.array([dJdW1, dJdW2, dJdW3])
        bias_gradients = np.array([dJdB1, dJdB2, dJdB3])

        return weight_gradients, bias_gradients

    def gradient_adjust(self, X, y, iterations=1000, learning_rate=0.5, reg_lambda=0.01, display=False, regularize=False):
        # train neural network

        if display:
            ax = plt.gca()
            ax.set_title('Gradient Descent', fontsize=28)
            plt.xlabel('number of iterations', fontsize=18)
            plt.ylabel('squared error', fontsize=18)

        for num in range(iterations):

            # generate mini batches
            X_batches, y_batches = sgd.mini_batch_generate(X, y, 10)

            for X_batch, y_batch in zip(X_batches, y_batches):

                # calculate gradients
                weight_gradients, bias_gradients = self.back_propagate(X_batch, y_batch)

                # train weights and biases

                if regularize:
                    for weight_scheme in self.weights:
                        for weight in weight_scheme:
                            weight += reg_lambda * weight

                for i, weight_gradient_scheme in enumerate(weight_gradients):
                    for j, weight_gradient in enumerate(weight_gradient_scheme):
                        self.weights[i][j] -= learning_rate * weight_gradient

                for i, bias_gradient_scheme in enumerate(bias_gradients):
                    for j, bias_gradient in enumerate(bias_gradient_scheme):
                        self.biases[i][j] -= learning_rate * bias_gradient

            if display:
                ax.scatter(num, self.squared_error(X, y))

        if display:
            plt.show()

    def train(self, X, y, iterations=1000, learning_rate=0.5, display=False, reg_lambda=0.01, regularize=False):
        # neural net training

        self.gradient_adjust(X, y, iterations=iterations, learning_rate=learning_rate, reg_lambda=reg_lambda,
                             display=display, regularize=regularize)

    def accuracy(self, X, y, string=False):
        # produces the accuracy of neural net

        yhat = self.predict(X)

        error_sum = np.sum(np.absolute(np.subtract(yhat, y)))
        y_sum = np.sum(np.absolute(y))
        accuracy = 1.0 - error_sum / y_sum

        if string:
            print("accuracy: " + str(accuracy * 100.0) + "%")
        else:
            return accuracy

    def squared_error(self, X, y):
        # produces the total squared error of neural net
        error = 0.0
        yHat = self.predict(X)
        for i in range(len(yHat)):
            error += ((y[i] - yHat[i]) ** 2) * 0.5

        return error

