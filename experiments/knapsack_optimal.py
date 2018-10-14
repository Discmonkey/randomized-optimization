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

from array import array

random = Random()
# The number of items
NUM_ITEMS = 10000
# The number of copies each
COPIES_EACH = 15
# The maximum weight for a single element
MAX_WEIGHT = 100
# The maximum volume for a single element
MAX_VOLUME = 100
# The volume of the knapsack
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME


# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)

initial_distribution = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mutation_function = DiscreteChangeOneMutation(ranges)

cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hill_climbing_problem = GenericHillClimbingProblem(ef, initial_distribution, nf)
genetic_problem = GenericGeneticAlgorithmProblem(ef, initial_distribution, mutation_function, cf)
probablistic_optimization = GenericProbabilisticOptimizationProblem(ef, initial_distribution, df)

from time import time
f = open("experiments/results/knapsack_optimal2.txt", "w")

f.write("starting RHC\n")
rhc = RandomizedHillClimbing(hill_climbing_problem)
score = 0
iters = 0
t0 = time()

while iters < 80000:
    score = rhc.train()
    f.write(str(iters) + "," + str(score) +"\n")
    iters += 1


print "RHC: " + str(ef.value(rhc.getOptimal())), "time taken", time() - t0, "Iterations:", iters

f.write("starting SA\n")
sa = SimulatedAnnealing(1E13, .95, hill_climbing_problem)
t0 = time()
iters = 0
score = 0

while iters < 80000:
    score = sa.train()
    f.write(str(iters) + "," + str(score) + "\n")
    iters += 1

print "SA: " + str(ef.value(sa.getOptimal())), "time taken", time() - t0, "Iterations", iters

ga = StandardGeneticAlgorithm(200, 100, 10, genetic_problem)
t0 = time()
iters = 0
score = 0

f.write("starting GA\n")
while iters < 5000:
    ga.train()
    score = ef.value(ga.getOptimal())
    f.write(str(iters) + "," + str(score) +"\n")
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
    f.write(str(iters) + "," + str(score) +"\n")
    iters += 1

print "MIMIC: " + str(ef.value(mimic.getOptimal())), "time taken", time() - t0, "Iterations", iters


