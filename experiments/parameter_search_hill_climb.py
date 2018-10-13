from __future__ import with_statement
"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing

TRAIN_FILE = os.path.join("datasets", "cache", "train_fire_reduced.csv")
TEST_FILE = os.path.join("datasets", "cache", "test_fire_reduced.csv")
WRITE_DIR = os.path.join("experiments", "results", "hill_climb_tests.txt")
INPUT_LAYER = 18
HIDDEN_LAYER_1 = 18

# training this network slightly differently since the example trains like a regression problem :(
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 10000


def initialize_instances(filename):
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []

    # Read in the abalone.txt CSV file
    with open(filename, "r") as abalone:
        reader = csv.reader(abalone)

        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            instance = Instance([float(value) for value in row[1:]])
            instance.setLabel(Instance(int(row[0])))
            instances.append(instance)

    return instances


def train(oa, network, oaName, instances, measure, fileobject):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    fileobject.write(str(oaName) + " training " + "\n")

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)
        print "finished iter", iteration, "for", oaName
        fileobject.write(str(oaName) + "," + str(iteration) + "," + str(error) + "\n")


def test(test_instances, network, oaname, fileobject):
    correct_negatives = 0
    incorrect_negatives = 0
    correct_positives = 0
    incorrect_positives = 0
    total = 0

    for instance in test_instances:
        network.setInputValues(instance.getData())
        network.run()

        output_values = network.getOutputValues()
        predicted = output_values.get(0)

        actual = instance.getLabel().getContinuous()

        result = 1 if predicted > .5 else 0
        actual = 1 if actual > .5 else 0

        if actual == 1 and result == 1:
            correct_positives += 1

        if actual == 1 and result == 0:
            incorrect_positives += 1

        if actual == 0 and result == 0:
            correct_negatives += 1

        if actual == 0 and result == 1:
            incorrect_negatives += 1

        total += 1

    fileobject.write(
        "||" + oaname + " testing, correct_positives, incorrect_postivies, correct_negatives, incorrect_negatives, total\n")
    fileobject.write(str(correct_positives) + "," + str(incorrect_positives) + ',' + str(correct_negatives)
                     + "," + str(incorrect_negatives) + ',' + str(total) + "\n")


def main():
    """Run algorithms on the abalone dataset."""
    train_instances = initialize_instances(TRAIN_FILE)
    test_instances = initialize_instances(TEST_FILE)

    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm

    oa_names = ["trial_1", "trial_2", "trial_3", "trail_4", "trial_5"]

    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER_1, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    for num, params in enumerate(oa_names):
        oa.append(RandomizedHillClimbing(nnop[num]))

    result_file = open(WRITE_DIR, "w")

    for i, name in enumerate(oa_names):
        start = time.time()

        train(oa[i], networks[i], oa_names[i], train_instances, measure, result_file)
        end = time.time()

        training_time = end - start

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        result_file.write("finished_training " + name + " in " + str(training_time))

        test(test_instances, networks[i], name, result_file)

        print "finished training, " + name

    result_file.close()


if __name__ == "__main__":
    main()

