import matplotlib.pyplot as plt


neural_net_performance = [81, 25, 60, 34]

peak_accuracy = 0
all_genetic_results = []
with open("/home/biometrics/BiometricsHg/sandbox/mgrinchenko/"
          "randomized-optimization/experiments/results/genetic_parameter_search.txt") as f:
    for line in f:
        if '||' in line:
            results = next(f)

            all_genetic_results.append(map(int, results.split(',')[0:4]))

print all_genetic_results

peak_accuracy = 0
all_annealing_results = []
with open("/home/biometrics/BiometricsHg/sandbox/mgrinchenko/randomized-optimization/"
          "experiments/results/simulated_annealing.txt") as f:
    for line in f:
        if '||' in line:
            results = next(f)

            all_annealing_results.append(map(int, results.split(',')[0:4]))

print all_annealing_results

peak_accuracy = 0
all_hillclimb_results = []
with open("/home/biometrics/BiometricsHg/sandbox/mgrinchenko/randomized-optimization/"
          "experiments/results/hill_climb_tests.txt") as f:
    for line in f:
        if '||' in line:
            results = next(f)

            all_hillclimb_results.append(map(int, results.split(',')[0:4]))

print all_hillclimb_results


def get_accuracy(some_list):
    return map(lambda pp: float(pp[0] + pp[2]) / 200, some_list)


def get_pos_recall(some_list):
    return map(lambda pp: float(pp[0]) / (pp[0] + pp[1]), some_list)


def get_neg_recall(some_list):
    return map(lambda pp: float(pp[2]) / (pp[2] + pp[3]), some_list)


def mean(some_list):
    return sum(some_list) / len(some_list)


figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_ylim([0, 1])
    
ax1.set_title("Peak accuracy")
ax1.axhline(float(neural_net_performance[0] + neural_net_performance[2]) / 200, label="Backprop", color='red')
ax1.bar(["HC", "SA", "GA"], [max(get_accuracy(res)) for res in [all_hillclimb_results, all_annealing_results,
                                                                all_genetic_results]])

ax2.set_title("Peak Positive Recall")
ax2.axhline(get_pos_recall([neural_net_performance])[0], label="Backprop", color='red')
ax2.bar(["HC", "SA", "GA"], [max(get_pos_recall(res)) for res in [all_hillclimb_results, all_annealing_results,
                                                                all_genetic_results]])

ax3.set_title("Peak Negative Recall")
ax3.axhline(get_neg_recall([neural_net_performance])[0], label="Backprop", color='red')
ax3.bar(["HC", "SA", "GA"], [max(get_neg_recall(res)) for res in [all_hillclimb_results, all_annealing_results,
                                                                  all_genetic_results]])

ax4.set_title("Avg accuracy")
ax4.axhline(float(neural_net_performance[0] + neural_net_performance[2]) / 200, label="Backprop", color='red')
ax4.bar(["HC", "SA", "GA"], [mean(get_accuracy(res)) for res in [all_hillclimb_results, all_annealing_results,
                                                                all_genetic_results]])

ax5.set_title("Avg Positive Recall")
ax5.axhline(get_pos_recall([neural_net_performance])[0], label="Backprop", color='red')
ax5.bar(["HC", "SA", "GA"], [mean(get_pos_recall(res)) for res in [all_hillclimb_results, all_annealing_results,
                                                                  all_genetic_results]])

ax6.set_title("Avg Negative Recall")
ax6.axhline(get_neg_recall([neural_net_performance])[0], label="Backprop", color='red')
ax6.bar(["HC", "SA", "GA"], [mean(get_neg_recall(res)) for res in [all_hillclimb_results, all_annealing_results,
                                                                  all_genetic_results]])


plt.show()
