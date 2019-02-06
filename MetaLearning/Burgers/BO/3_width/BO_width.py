"""
@author: Maziar Raissi
"""


import os
import tensorflow as tf
import numpy as np
import scipy.io
import csv
from bayes_opt import BayesianOptimization
from pyDOE import lhs
from datetime import datetime



import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# don't fix seed every time, it actually will incorporate noise in y
np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, max_iter=50000):

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers
        self.nu = nu

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session()

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': max_iter,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

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

# trace the process in each iteration


def trace(func):
    num = 0
    run_time = datetime.now() - datetime.now()
    err = []

    def call_func(node_per_layer, num_layer, data, max_train_iter):
        nonlocal num
        nonlocal run_time
        num += 1
        start_time = datetime.now()
        error = func(node_per_layer, num_layer, data, max_train_iter)
        err.append(error)
        end_time = datetime.now()
        run_time += end_time - start_time
        if (num-1) % 50 == 0 and num != 1:
            h, temp = divmod((run_time - (end_time - start_time)).total_seconds(), 3600)
            m, s = divmod(temp, 60)
            best_value = max(err)
            print('-' * 100)
            print('\033[0;33mTime of %d iterations = %dh%dm%ds\033[0m' % (num-1, h, m, s))
            print('\033[0;33mBest results: %.5f\033[0m' % best_value)
            print('-' * 100)
        out = open('output_width.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([node_per_layer, num_layer, error])
        out.close()
        return error
    return call_func

@trace
def eval_NN_performance(node_per_layer, num_layer, data, max_train_iter):
    # setting of NN Arch.
    start_time = datetime.now()
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

    '''
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    print idx
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
    '''

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, max_train_iter)
    model.train()

    # training error
    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    end_time = datetime.now()
    m, s = divmod((end_time - start_time).total_seconds(), 60)

    return -error_u


if __name__ == "__main__":
    data = scipy.io.loadmat('./burgers_shock.mat')
    max_iter = 5000
    num_layer = 10
    print('num_layer = 10')
    bo = BayesianOptimization(
        lambda node_per_layer: eval_NN_performance(node_per_layer, num_layer, data, max_iter),
        {'node_per_layer': (2, 100)})
    # gp_params = {'kernel': None,
    #             'alpha': 1e-5}

    # bo.maximize(init_points=5, acq='ei', xi=0.1, **gp_params)
    # bo.maximize(init_points=5, acq='ei', xi=0.1, **gp_params)
    bo.maximize(init_points=2, n_iter=48, acq="ei", xi=1e-4)
    print(bo.res['max'])
    node_per_layer_star = bo.res['max']['max_params']['node_per_layer']
    num_layer_star = bo.res['max']['max_params']['num_layer']
    print(eval_NN_performance(node_per_layer_star, num_layer_star, data, 50000))


