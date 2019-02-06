import tensorflow as tf
import numpy as np
from SALib.sample import sobol_sequence
import sys
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main_fun(num_try=1, width=20, depth=4, iter_num=50000, learning_rate=5.0e-4, Ntr=100):
    def u_exact(x, t):
        u = np.sin(np.pi * x) * np.exp(-t)
        return u

    def f_exact(x, t):
        f = -1.0 * np.exp(-t) * np.sin(np.pi * x) \
            + np.pi ** 2 * np.sin(np.pi * x) * np.exp(-t)
        return f

    layers = [2] + [width] * depth + [1]
    L = len(layers)

    ####  randomaly sampled points
    space_dim = 1
    #Ntr = 100

    xt_f = sobol_sequence.sample(Ntr, space_dim + 1)[1:, :]
    x_f = np.reshape(xt_f[:, 0], [-1, 1])
    t_f = np.reshape(xt_f[:, 1], [-1, 1])
    x_f = -1.0 + 2.0 * x_f
    f_value = f_exact(x_f, t_f)

    def g_fun(x, t, u):
        return t * (-1.0 - x) * (1.0 - x) * u + tf.sin(np.pi * x)

    def xavier_init(size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    def neural_net(X, weights, biases):
        num_layers = len(weights) + 1
        H = X  # 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(x_u, t_u, weights, biases):
        u = neural_net(tf.concat([x_u, t_u], 1), weights, biases)
        return u

    def net_f(x_f, t_f, weights, biases):

        u = g_fun(x_f, t_f, net_u(x_f, t_f, weights, biases))
        u_x = tf.gradients(u, x_f)[0]
        u_xx = tf.gradients(u_x, x_f)[0]
        u_t = tf.gradients(u, t_f)[0]

        return u_t - u_xx

    f_target = tf.to_double(np.reshape(f_value, [-1, 1]))
    x_f_tf = tf.to_double(tf.reshape(x_f, [-1, 1]))
    t_f_tf = tf.to_double(tf.reshape(t_f, [-1, 1]))

    x_test = x_f
    t_test = t_f
    x_test_tf = x_f_tf
    t_test_tf = t_f_tf
    ut = u_exact(x_test, t_test)

    loss_vec = np.zeros((num_try, 1), dtype=np.float64)
    error_vec = np.zeros((num_try, 1), dtype=np.float64)
    loss_mat = []
    error_u_mat = []

    for num_run in range(num_try):
        min_loss = 1.0e16
        loss_record = []
        error_u_record = []

        weights = [xavier_init([layers[l], layers[l + 1]]) for l in range(0, L - 1)]
        biases = [tf.Variable(tf.zeros((1, layers[l + 1]), dtype=tf.float64), trainable=True) for l in range(0, L - 1)]

        f_pred = net_f(x_f_tf, t_f_tf, weights, biases)

        loss = tf.reduce_mean(tf.square(f_target - f_pred))  # \

        optimizer_adam = tf.train.AdamOptimizer(learning_rate)
        train_op_adam = optimizer_adam.minimize(loss)
        error_u_opt = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            uu = g_fun(x_test_tf, t_test_tf, net_u(x_test_tf, t_test_tf, weights, biases))
            for i in range(iter_num):
                sess.run(train_op_adam)
                if i % 1000 == 0:
                    temp_loss = sess.run(loss)
                    if temp_loss < min_loss:
                        min_loss = temp_loss
                        ut_opt = np.reshape(sess.run(uu), [-1, 1])
                        error_u_opt = np.linalg.norm(ut_opt - ut, 2) / np.linalg.norm(ut, 2)
                    loss_record.append(temp_loss)
                    ut0 = np.reshape(sess.run(uu), [-1, 1])
                    error_u0 = np.linalg.norm(ut0 - ut, 2) / np.linalg.norm(ut, 2)
                    error_u_record.append(error_u0)

        loss_vec[num_run] = min_loss

        error_vec[num_run] = error_u_opt

        loss_mat.append(np.reshape(loss_record, [-1, 1]))
        error_u_mat.append(np.reshape(error_u_record, [-1, 1]))

    return np.mean(error_vec)


if __name__ == "__main__":

    node_per_layer = 41
    num_layer = 3
    Ntr = np.arange(50, 450, 50)
    for i in Ntr:
        error = list([])
        for j in range(10):
            np.random.seed(i+j)
            cur_err = main_fun(num_try=1, width=node_per_layer, depth=num_layer, 
                               iter_num=50000, learning_rate=5.0e-4, Ntr=i)
            error.append(cur_err)
        error = np.array(error)
        mean = float(np.mean(error))
        std = float(np.std(error))
        # print('[node_per_layer,num_layer,mean,std]')
        print([i, mean, std])
        # print('print over!')
