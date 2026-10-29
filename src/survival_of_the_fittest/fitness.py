import numpy as np


def fitness_function(individual, target_sentence="Hello World"):
    assert len(individual) == len(target_sentence)

    score = np.sum([
        individual[i] == target_sentence[i]
        for i in range(len(target_sentence))
    ])

    return score
