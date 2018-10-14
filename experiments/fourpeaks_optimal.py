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
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
from array import array

"""
Commandline parameter(s):
   none
"""

N= 10000
T= N / 4
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)

initial_distribution = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mutation_function = DiscreteChangeOneMutation(ranges)

cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hill_climbing_problem = GenericHillClimbingProblem(ef, initial_distribution, nf)
genetic_problem = GenericGeneticAlgorithmProblem(ef, initial_distribution, mutation_function, cf)
probablistic_optimization = GenericProbabilisticOptimizationProblem(ef, initial_distribution, df)

from time import time
f = open("experiments/results/fourpeaks_optimal.txt", "w")

f.write("starting RHC\n")
rhc = RandomizedHillClimbing(hill_climbing_problem)
score = 0
iters = 0
t0 = time()

while iters < 60000:
    score = rhc.train()
    f.write(str(iters) + str(score))
    iters += 1


print "RHC: " + str(ef.value(rhc.getOptimal())), "time taken", time() - t0, "Iterations:", iters

f.write("starting SA\n")
sa = SimulatedAnnealing(1E13, .95, hill_climbing_problem)
t0 = time()
iters = 0
score = 0

while iters < 60000:
    score = sa.train()
    f.write(str(iters) + str(score))
    iters += 1

print "SA: " + str(ef.value(sa.getOptimal())), "time taken", time() - t0, "Iterations", iters

ga = StandardGeneticAlgorithm(200, 100, 10, genetic_problem)
t0 = time()
iters = 0
score = 0

f.write("starting GA\n")
while iters < 20000:
    ga.train()
    score = ef.value(ga.getOptimal())
    f.write(str(iters) + str(score))
    iters += 1

print "GA: " + str(ef.value(ga.getOptimal())), "time taken", time() - t0, "Iterations", iters

mimic = MIMIC(200, 100, probablistic_optimization)
score = 0
t0 = time()
iters = 0

f.write("starting MIMIC\n")
while iters < 1000:
    mimic.train()
    score = ef.value(mimic.getOptimal())
    print iters, score
    f.write(str(iters) + str(score))
    iters += 1

print "MIMIC: " + str(ef.value(mimic.getOptimal())), "time taken", time() - t0, "Iterations", iters


