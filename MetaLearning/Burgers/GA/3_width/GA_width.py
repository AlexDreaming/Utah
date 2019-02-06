"""
@author: Maziar Raissi
"""

import sys
import os
import csv
import tensorflow as tf
import numpy as np
from pyDOE import lhs
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# don't fix seed every time, it actually will incorporate noise in y

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, X_star, u_star, layers, lb, ub, nu, max_iter=500):

        self.lb = lb
        self.ub = ub

        self.X_star = X_star
        self.u_star = u_star

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u
        self.max_iter = max_iter

        self.layers = layers
        self.nu = nu

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'gpu': 0}))

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.f_err = tf.reduce_mean(tf.square(self.f_pred))

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + self.f_err

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': max_iter,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        # using Adam
        # opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #opt = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate)
        # self.optimizer = opt.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        # print self.sess.run(self.weights)

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

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, seed=1234), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u * u_x - self.nu * u_xx

        return f

    def callback(self, loss):
        pass
        # print('Loss:', loss)

    def train(self):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star


def trace(func):
    num = 0
    run_time = datetime.now() - datetime.now()
    err = []

    def call_func(node_per_layer, num_layer, data, max_train_iter):
        nonlocal num
        nonlocal run_time
        num += 1
        print("Step:", num)
        error, time = func(node_per_layer, num_layer, data, max_train_iter)
        err.append(error)
        run_time += time
        if num % 50 == 0:
            h, temp = divmod(run_time.total_seconds(), 3600)
            m, s = divmod(temp, 60)
            best_value = min(err)
            print('-' * 100)
            print('\033[0;33mTime of %d iterations = %dh%dm%ds\033[0m' % (num, h, m, s))
            print('\033[0;33mBest results: %.5f\033[0m' % best_value)
        print('-' * 100)
        return float(error)
    return call_func

@trace
def eval_NN_performance(node_per_layer, num_layer, data, max_train_iter):
    np.random.seed(1234)
    tf.set_random_seed(1234)

    # setting of NN Arch.
    node_per_layer = int(np.round(node_per_layer))
    num_layer = int(np.round(num_layer))
    layers = [2] + [node_per_layer] * num_layer + [1]
    # training
    nu = 0.01 / np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    start_time = datetime.now()
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train,  X_star, u_star, layers, lb, ub, nu, max_train_iter)
    # model.X_star = X_star
    # model.u_star = u_star

    model.train()

    # training error
    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    end_time = datetime.now()
    m, s = divmod((end_time - start_time).total_seconds(), 60)
    if error_u >= 0.001:
        print('node_per_layer = %d, | num_layer = %d | value = %g | time = %dm%ds' %
              (node_per_layer, num_layer, error_u, m, s))
    else:
        print('\033[0;35mnode_per_layer = %d, | num_layer = %d | value = %g | time = %dm%ds \033[0m' %
              (node_per_layer, num_layer, error_u, m, s))
    f = open("./width_cmd.txt", "a")
    f.write('python run_GA.py %d %d\n' % (node_per_layer, num_layer))
    return float(error_u), end_time - start_time


