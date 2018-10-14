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

    plt.plot(results["100,100,50 training"], label="100 parents selected for reproduction")
    plt.plot(results["100,50,50 training"], label="50 parents selected for reproduction")
    plt.title("Quantifying effect of 'population age'")

    plt.legend()
    plt.ylabel("L2 Loss")
    plt.xlabel("Iterations")
    plt.show()
