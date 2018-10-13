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

    for key, array in results.items():
        plt.plot(array, label=key)
    #
    plt.legend()
    plt.title("Simulated Annealing Tuning and Parameter Search")
    plt.ylabel("L2 Loss")
    plt.xlabel("Iterations")
    plt.show()
