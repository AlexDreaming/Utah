"""
    hyperband.py

    `hyperband` algorithm for hyper-parameter optimization

    Copied/adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

    The algorithm is motivated by the idea that random search is pretty
    good as far as black-box optimization goes, so let's try to do it faster.
"""

import sys
import numpy as np
from math import log, ceil
from datetime import datetime


class HyperBand:

    def __init__(self, model, max_iter=1, eta=3):

        self.model = model

        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(log(max_iter) / log(eta))
        self.B = (self.s_max + 1) * max_iter

        self.best_obj = np.inf
        self.history = []

    def run(self):
        for s in reversed(range(self.s_max + 1)):

            # initial number of configs
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # initial configs
            configs = [self.model.rand_config() for _ in range(n)]
            print(configs)
            f = open("./dw_cmd.txt", "a")
            for config in configs:
                f.write('python run_HB.py %d %d\n' % (config['node_per_layer'], config['num_layer']))
            f.close()
            for i in range(s + 1):

                # number of iterations for these configs
                r_i = r * self.eta ** i
                print("\n -- %d configs  -- \n" % (len(configs)))
                sys.stdout.flush()

                results = []
                for j, config in enumerate(configs):
                    print("Config %d: %s" % (j+1, config))
                    sys.stdout.flush()

                    start_time = datetime.now()
                    res = self.model.eval_config(config=config)
                    results.append(res)
                    end_time = datetime.now()
                    m, s = divmod((end_time - start_time).total_seconds(), 60)
                    self.best_obj = min(res['obj'], self.best_obj)
                    print("Current: %f | Best: %f | Time: %dm%ds" % (float(res['obj']), self.best_obj, m,s))
                    print('-'*100)
                    # if (j+1) % 50 == 0:
                    #     h, temp = divmod((end_time - init_time).total_seconds(), 3600)
                    #     m, s = divmod(temp, 60)
                    #     print('\033[0;33mTime of %d iterations = %dh%dm%ds\033[0m' % (j+1, h, m, s))
                    #     print('-' * 100)
                    sys.stdout.flush()

                    #print(res)
                    sys.stdout.flush()

                self.history += results

                # Sort by objective value
                results = sorted(results, key=lambda x: x['obj'])

                # Drop models that have already converged
                results = list(filter(lambda x: not x.get('converged', False), results))
                # Determine how many configs to keep
                n_keep = int(n * self.eta ** (-i - 1))
                results = results[:n_keep]
                configs = [result['config'] for result in results]

        return results
