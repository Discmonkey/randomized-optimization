import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
# crossovers
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.ga.SingleCrossOver as SingleCrossOver


import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer

from array import array



"""
Commandline parameter(s):
   none
"""

N= 1000
T= N / 4
fill = [2] * N
ranges = array('i', fill)

ef = CountOnesEvaluationFunction()
initial_distribution = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mutation_function = DiscreteChangeOneMutation(ranges)

cf = SingleCrossOver()


df = DiscreteDependencyTree(.1, ranges)
hill_climing_problem = GenericHillClimbingProblem(ef, initial_distribution, nf)
genetic_problem = GenericGeneticAlgorithmProblem(ef, initial_distribution, mutation_function, cf)
probablistic_optimization = GenericProbabilisticOptimizationProblem(ef, initial_distribution, df)

from time import time


# rhc = RandomizedHillClimbing(hcp)
# score = 0
# iters = 0
# t0 = time()
#
# while score < N:
#     print score
#     score = rhc.train()
#     iters += 1
#
#
# print "RHC: " + str(ef.value(rhc.getOptimal())), "time taken", time() - t0, "Iterations:", iters
#
# sa = SimulatedAnnealing(1E11, .95, hcp)
# t0 = time()
# iters = 0
# score = 0
#
# while score < N:
#     print score
#     score = sa.train()
#     iters += 1
#
# print "SA: " + str(ef.value(sa.getOptimal())), "time taken", time() - t0, "Iterations", iters

ga = StandardGeneticAlgorithm(1000, 30, 1, genetic_problem)
t0 = time()
iters = 0
score = 0

while score < N and iters < 10000:
    ga.train()
    score = ef.value(ga.getOptimal())
    print score
    iters += 1

print "GA: " + str(ef.value(ga.getOptimal())), "time taken", time() - t0, "Iterations", iters

mimic = MIMIC(200, 100, probablistic_optimization)
score = 0
t0 = time()
iters = 0

while score < N:
    mimic.train()
    score = ef.value(mimic.getOptimal())
    print score
    iters += 1

print "MIMIC: " + str(ef.value(mimic.getOptimal())), "time taken", time() - t0, "Iterations", iters


