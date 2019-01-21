#! python3
import collections
import random

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Model:
    def __init__(self, layers):
        self.weight = [2 * np.random.random((l1, l2)) - 1
                       for l1, l2 in zip(layers, layers[1:])]

    def forward(self, x):
        layer = [x]
        for weight in self.weight:
            layer.append(sigmoid(np.dot(layer[-1], weight)))
        return layer

    def train(self, x, y, epochs):
        weight = self.weight
        lw = len(weight)
        forward = self.forward
        for i in range(epochs):
            layer = forward(x)

            error = [y - layer[-1]]
            delta = [error[0] * sigmoid_derivative(layer[-1])]
            if not i or i == epochs - 1:
                print(f"Error: {np.mean(np.abs(error[0]))}")

            for i in range(1, lw):
                error.insert(0, delta[0].dot(weight[-i].T))
                delta.insert(0, error[0] * sigmoid_derivative(layer[-i - 1]))

            for i in range(lw):
                weight[i] += layer[i].T.dot(delta[i])

    def save(self, name):
        for a, w in enumerate(self.weight):
            np.save(f"{name}_{a}.npy", w)

    def load(self, name):
        self.weight = [np.load(f"{name}_{a}.npy")
                       for a in range(len(self.weight))]


def main():
    epochs = 1000
    layers = [10, 90, 30, 3]
    model = Model(layers)

    t = 0.5  # Tie value
    correctChoice = [np.array([[t, 1, 0]]),
                     np.array([[0, t, 1]]),
                     np.array([[1, 0, t]])]

    counter = [0, 0, 0, 0]
    last = collections.deque((i % 3 for i in range(10)), maxlen=layers[0])
    x = np.array([last])

    helpText = """
    0.Exit
    1. Rock
    2. Paper
    3. Scissors

    h -> help
    q -> quit
    s -> save
    l -> load
    """
    numbers = {"1": 0, "2": 1, "3": 2}
    names = ("rock", "paper", "scissors")
    outcomes = ("Win", "Tie", "Lose")
    EXIT = {"0", "q", "quit", "exit"}

    print(helpText)

    while True:
        ignore = True
        try:
            option = input("> ")
        except KeyboardInterrupt:
            option = "exit"

        if option in EXIT:
            print("Goodbye!")
            break
        elif option == "h" or option == "help":
            print(helpText)
        elif option == "s" or option == "save":
            model.save("save")
        elif option == "l" or option == "load":
            model.load("save")
        elif option not in numbers:
            print("That was not 1, 2 or 3")
        else:
            ignore = False

        if ignore:
            continue

        number = numbers[option]
        predictions = model.forward(x)[-1][0]
        machine = max(enumerate(predictions), key=lambda x: x[1])[0]

        if (machine + 1) % 3 == number % 3:
            o = 0
            print("[%8s] vs  %8s " % (names[number], names[machine]))
        elif machine == number:
            o = 1
            print(" %8s  vs  %8s " % (names[number], names[machine]))
        else:
            o = 2
            print(" %8s  vs [%8s]" % (names[number], names[machine]))
        counter[o] += 1
        counter[3] += 1
        print("%4s   %03d | %03d | %03d (w|t|l) %.2f | %.2f | %.2f" %
              (outcomes[o], *counter[:3],
               counter[0] * 100 // counter[3],
               counter[1] * 100 // counter[3],
               counter[2] * 100 // counter[3]))

        if counter[3] >= layers[0]:
            y = correctChoice[number]
            model.train(x, y, epochs)

        last.append(number)
        last.append(machine)
        x = np.array([last])


if __name__ == "__main__":
    main()
