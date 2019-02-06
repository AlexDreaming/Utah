#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
from datetime import datetime
from GA_dw import eval_NN_performance
from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

data = scipy.io.loadmat('./burgers_shock.mat')
max_train_iter = 2

# Define population.
indv_template = BinaryIndividual(ranges=[(2, 50), (2, 20)], eps=0.001)
population = Population(indv_template=indv_template, size=18).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])


# Define fitness function.
@engine.fitness_register
def fitness(indv):
    node_per_layer, num_layer = indv.solution
    value = -eval_NN_performance(node_per_layer, num_layer, data, max_train_iter)
    return value


@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):

    interval = 1
    master_only = True
    
    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        print('\033[0;31mGeneration: {}\033[0m'.format(g+1))
        print('-' * 100)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        print('\033[0;31Optimal solution: (node_per_layer:{}, num_layer:{}, value:{})\033[0m'.format(x[0], x[1], y))


if '__main__' == __name__:
    start = datetime.now()
    engine.run(ng=5)
    end = datetime.now()
    h, temp = divmod((end - start).total_seconds(), 3600)
    m, s = divmod(temp, 60)
    print('\033[0;31mTotal Time: %dh%dm%ds\033[0m' % (h, m, s))
