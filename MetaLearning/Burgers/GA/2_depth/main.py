#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
from datetime import datetime
from GA_depth import eval_NN_performance
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
max_train_iter = 5000

# Define population.
indv_template = BinaryIndividual(ranges=[(1, 40)], eps=0.001)
population = Population(indv_template=indv_template, size=10).init()

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
    node_per_layer = 10
    num_layer = indv.solution
    value = -eval_NN_performance(node_per_layer, num_layer, data, max_train_iter)
    return value


@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):

    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        pass

    def finalize(self, population, engine):
        pass


if '__main__' == __name__:
    engine.run(ng=4)
