

import os
import tensorflow as tf
import numpy as np
import scipy.io
import csv
import random


from pyDOE import lhs
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# don't fix seed every time, it actually will incorporate noise in y

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True


if __name__ == "__main__":
    #num_layer = 10
    #node_per_layer = 10
    # Random Search
    for i in range(100):
        np.random.seed(i)
        num_layer = random.choice(range(2, 21))
        #node_per_layer = random.choice(range(20, 61))
        node_per_layer = random.choice(range(2, 51))
        #f = open("./dw_cmd.txt", "a")
        #f.write('python run_RS.py %d %d\n' % (node_per_layer, num_layer))
        print('python run_RS.py %d %d' % (node_per_layer, num_layer))

