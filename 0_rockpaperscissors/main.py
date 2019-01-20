#! python3
import numpy as np

# Learnt here -> https://iamtrask.github.io/2015/07/12/basic-python-network/
OPTIONS = set("123")
NAMES = ("rock", "paper", "scissors")
EXIT = {"0", "q", "exit"}


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def create(layers):
    return [2 * np.random.random((layers[a], layer)) - 1
            for a, layer in enumerate(layers[1:])]


def train(x, y, weights, epochs):
    for i in range(epochs):
        layer = [x]
        for weight in weights:
            layer.append(nonlin(np.dot(layer[-1], weight)))

        error = [y - layer[-1]]
        delta = [error[0] * nonlin(layer[-1], deriv=True)]
        if not i or i == epochs - 1:
            print(f"Error: {np.mean(np.abs(error[0]))}")

        for i in range(1, len(weights)):
            error.insert(0, delta[0].dot(weights[-i].T))
            delta.insert(0, error[0] * nonlin(layer[-i - 1], deriv=True))
        weights = [weight + layer[a].T.dot(delta[a])
                   for a, weight in enumerate(weights)]
    return weights


def test(x, weights):
    layer = [x]
    for weight in weights:
        layer.append(nonlin(np.dot(layer[-1], weight)))
    return layer[-1]


def main():
    print("0.Exit\n1. Rock\n2. Paper\n3. Scissors\n\n")
    layers = [3, 90, 3]
    weights = create(layers)
    counter = [0, 0, 0]
    last = [i % 3 for i in range(10)]
    x = np.array([last[-layers[0]:]])
    while True:
        option = input("> ")
        if option in EXIT:
            print("Goodbye!")
            break
        if option == "save":
            for a, w in enumerate(weights):
                np.save(f"save_{a}.npy", w)
            continue
        elif option == "load":
            weights = [np.load(f"save_{a}.npy") for a in range(len(weights))]
            continue
        if option not in OPTIONS:
            print("That was not between 1 and 3 included")
            continue
        option = int(option) - 1
        last.append(option)

        machine = max((i, a) for a, i in enumerate(test(x, weights)[0]))[1]
        print("[%8s] vs [%8s]" % (NAMES[option], NAMES[machine]))
        if (machine + 1) % 3 == option % 3:
            s = "Win"
            counter[0] += 1
        elif machine == option:
            s = "Tie"
            counter[1] += 1
        else:
            s = "Lose"
            counter[2] += 1
        print("%4s   %d | %d | %d (w|t|l)" % (s, *counter))
        y = np.array([[0 if ((i + 1) % 3 == option % 3) else
                       (0.5 if i == option else 1) for i in range(3)]])
        weights = train(x, y, weights, 10000)
        x = np.array([last[-layers[0]:]])


if __name__ == "__main__":
    main()
