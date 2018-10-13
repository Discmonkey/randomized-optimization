from parser_args import get_parser
import matplotlib.pyplot as plt


def parse_training_file(filepath):
    training_dict = {

    }
    current = None
    skip_next = False
    count = 0
    with open(filepath) as file:
        for line in file:
            if "||" in line:
                skip_next = True
                continue

            if "training" in line:
                training_dict[line.strip()] = []
                current = training_dict[line.strip()]
                count = 0
                continue

            if skip_next:
                skip_next = False
                continue

            if count > 1000:
                current.append(float(line.split(',').pop().strip()))

            count += 1

    for key in training_dict.keys():
        if ".97" in key:
            training_dict.pop(key)

    return training_dict


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    file_path = args.filepath

    results = parse_training_file(file_path)

    num_graphs = len(results.keys())

    figsize, (ax1) = plt.subplots(1, 1, sharex=True)

    fig_size = plt.rcParams["figure.figsize"]

    fig_size[1] = 20
    fig_size[0] = 8
    ax1.set_title("Convergence Across Temperature Decay for T=1E9")
    # ax2.set_title("1E12 Starting Temperature")
    # ax3.set_title("1E15 Starting Temperature")

    for key, array in results.items():
        if "1000000000" in key:
            ax1.plot(array, label=key.split(',').pop().replace("training", "").strip())

        # elif "1e+12" in key:
        #     ax2.plot(array, label=key.split(',').pop().replace("training", "").strip())
        #
        # elif "1e+15" in key:
        #     ax3.plot(array, label=key.split(',').pop().replace("training", "").strip())
    ax1.legend()
    # ax2.legend()
    # ax3.legend()

    plt.legend()
    plt.ylabel("L2 Loss")
    plt.xlabel("Iterations past 1000")
    plt.show()

