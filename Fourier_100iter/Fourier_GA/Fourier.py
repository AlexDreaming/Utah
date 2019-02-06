import os
import tensorflow as tf
import numpy as np
import scipy.io
import csv
from scipy.interpolate import griddata
from pyDOE import lhs
from datetime import datetime

from scipy.special import gamma
from scipy.special import jv

import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# don't fix seed every time, it actually will incorporate noise in y
np.random.seed(1234)
tf.set_random_seed(1234)



class Nerwork:
    # Initialize the class
    def __init__(self, x, y, layers, iter=10000):

        self.x = x
        self.y = y
        self.iter = iter
        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        # self.learning_rate = lr

        # tf placeholders and graph
        self.sess = tf.Session()
        self.x_train = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_train = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])


        self.y_pred = self.net_output(self.x_train)

        self.loss = tf.reduce_mean(tf.square(self.y_train - self.y_pred))

        # using Adam
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = opt.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, seed=1234), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = X #2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_output(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u

    def train(self, learning_rate):

        # start_time = time.time()
        for it in range(self.iter):
            tf_dict = {self.x_train: self.x, self.y_train: self.y, self.learning_rate: learning_rate}

            _, train_loss, lr = self.sess.run(fetches=[self.optimizer, self.loss, self.learning_rate],
                                              feed_dict=tf_dict)


    def predict(self, x_test):
        y_pred = self.sess.run(self.y_pred, {self.x_train: x_test})
        return y_pred

def trace(func):
    num = 0

    def call_func(node_per_layer, num_layer, x_train, y_train, x_test, y_test, lr, max_train_iter):
        nonlocal num
        num += 1
        print("Step:", num)
        error = func(node_per_layer, num_layer, x_train, y_train, x_test, y_test, lr, max_train_iter)
        print('-' * 100)
        return float(error)
    return call_func

@trace
def eval_performance(node_per_layer, num_layer, x_train, y_train, x_test, y_test, lr, max_train_iter):
    start_time = datetime.now()
    np.random.seed(1234)
    tf.set_random_seed(1234)

    node_per_layer = int(np.round(node_per_layer))
    num_layer = int(np.round(num_layer))
    layers = [10] + [node_per_layer] * num_layer + [1]

    model = Nerwork(x_train, y_train, layers, max_train_iter)
    model.train(lr)

    # training error
    y_pred = model.predict(x_test)
    error_u = np.linalg.norm(y_test - y_pred, 2) / np.linalg.norm(y_test, 2)
    end_time = datetime.now()
    m, s = divmod((end_time - start_time).total_seconds(), 60)
    print('node_per_layer = %d, | num_layer = %d | value = %g | time = %dm%ds' %
          (node_per_layer, num_layer, error_u, m, s))
    return -error_u


def fourier_transform(x, d):
    rd = np.sqrt(1.0/np.pi) * np.power(gamma(d/2.0 + 1), 1.0/d)
    norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
    fourier = np.power(rd/norm, d/2.0) * jv(d/2.0, 2*np.pi*rd*norm)
    return fourier





