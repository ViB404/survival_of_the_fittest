import string

import numpy as np
from numpy import random


class GeneticAlgorithm:
    def __init__(self, num_attributes = 2, population_size: int = 100, population = None) -> None:
        self.population_size = population_size
        self.population = None
        self.num_attributes = num_attributes

    def initialize_population(self):
        attributes = []

        characters = list(string.ascii_uppercase + string.punctuation + string.whitespace)

        for _ in range(self.num_attributes):
            attributes.append(random.choice(characters, size=self.population_size))

        self.population = np.array(attributes).T