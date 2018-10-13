from parser_args import get_parser
import matplotlib.pyplot as plt


def parse_training_file(filepath):
    training_dict = {

    }
    current = None
    skip_next = False
    with open(filepath) as file:
        for line in file:
            if "||" in line:
                skip_next = True
                continue

            if "training" in line:
                training_dict[line.strip()] = []
                current = training_dict[line.strip()]
                continue



            if skip_next:
                skip_next = False
                continue

            current.append(float(line.split(',').pop().strip()))


    return training_dict


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    file_path = args.filepath

    results = parse_training_file(file_path)

    num_graphs = len(results.keys())

    figsize, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    fig_size = plt.rcParams["figure.figsize"]

    fig_size[1] = 20
    fig_size[0] = 8
    ax1.set_title("Population size 100")
    ax2.set_title("Population size 200")
    ax3.set_title("Population size 400")

    for key, array in results.items():
        if key.startswith("100"):
            params = key.split(',')
            parent_size = params[1]
            mutation_size = params[2].replace("training", "").strip()
            ax1.plot(array, label="Parents: {} Mutations: {}".format(parent_size, mutation_size))

        elif key.startswith("200"):
            params = key.split(',')
            parent_size = params[1]
            mutation_size = params[2].replace("training", "").strip()
            ax2.plot(array, label="Parents: {} Mutations: {}".format(parent_size, mutation_size))

        else:
            params = key.split(',')
            parent_size = params[1]
            mutation_size = params[2].replace("training", "").strip()
            ax3.plot(array, label="Parents: {} Mutations: {}".format(parent_size, mutation_size))

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.legend()
    plt.ylabel("L2 Loss")
    plt.xlabel("Iterations")
    plt.show()
