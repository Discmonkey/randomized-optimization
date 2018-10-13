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
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
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
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

from time import time


rhc = RandomizedHillClimbing(hcp)
fit = FixedIterationTrainer(rhc, 600000)
t0 = time()
fit.train()
print "RHC: " + str(ef.value(rhc.getOptimal())), "time taken", time() - t0

sa = SimulatedAnnealing(1E11, .95, hcp)
fit = FixedIterationTrainer(sa, 600000)

t0 = time()
fit.train()
print "SA: " + str(ef.value(sa.getOptimal())), "time taken", time() - t0

ga = StandardGeneticAlgorithm(200, 100, 10, gap)
fit = FixedIterationTrainer(ga, 20000)

t0 = time()
fit.train()

print "GA: " + str(ef.value(ga.getOptimal())), "time taken", time() - t0

mimic = MIMIC(50, 10, pop)
fit = FixedIterationTrainer(mimic, 50)

t0 = time()
fit.train()

print "MIMIC: " + str(ef.value(mimic.getOptimal())), "time taken", time() - t0
