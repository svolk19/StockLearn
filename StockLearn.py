import neural_net
import csv_reader as cr
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import time


def stock_data(NN, filepath):
    labels, data = cr.csv_reader(filepath)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, test_size=0.2)

    X_train = np.array(X_train).reshape((53, 3))
    X_test = np.array(X_test).reshape((14, 3))
    y_train = np.array(y_train).reshape((53, 1))
    y_test = np.array(y_test).reshape((14, 1))

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    y_train = preprocessing.normalize(y_train)
    y_test = preprocessing.normalize(y_test)

    start_time = time.time()
    NN.train(X_train, y_train, iterations=1000, learning_rate=0.01, regularize=False, display=True)
    end_time = time.time()

    total_time = end_time - start_time
    print(NN.accuracy(X_test, y_test), 'total time:', total_time)

if __name__ == '__main__':
    NN = neural_net.NeuralNetwork(3, 1, 10, 10)
    stock_data(NN, "C:/Users/sam/Desktop/HistoricalQuotes.csv")
