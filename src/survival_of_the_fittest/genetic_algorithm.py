import random
import string

import numpy as np

from survival_of_the_fittest.fitness import fitness_function


class GeneticAlgorithm:
    def __init__(self, num_attributes=11, population_size: int = 100, population=None,
                 fitness_function=fitness_function, cross_prob=0.75, mut_prob=0.05) -> None:
        self.population_size = population_size
        self.num_attributes = num_attributes
        self.fitness_function = fitness_function
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

        self.population = population
        self.fitness_scores = None

        self.characters = list(string.ascii_uppercase + string.whitespace + string.punctuation + string.ascii_lowercase)

    def initialize_population(self):
        attributes = []
        for _ in range(self.num_attributes):
            attributes.append(np.random.choice(self.characters, size=self.population_size))

        self.population = np.array(attributes).T

    def compute_fitness_score(self):
        scores = np.array([self.fitness_function(individual) for individual in self.population])
        self.fitness_scores = scores

    def selection_probability(self):
        abs_scores = np.abs(self.fitness_scores)
        total_sum = np.sum(abs_scores)

        if total_sum == 0:
            probability = np.ones(self.population_size) / self.population_size
        else:
            probability = abs_scores / total_sum

        parents = random.choices(self.population, weights=probability, k=2)
        return parents

    def run(self):
        def crossover(parents):
            if np.random.uniform() < self.cross_prob:
                parent_1, parent_2 = parents
                crossover_point = np.random.choice(range(1, self.num_attributes))
                child = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
                return child
            else:
                return random.choice(parents)

        def mutate(individual):
            new_individual = []
            for attribute in individual:
                if np.random.uniform() < self.mut_prob:
                    new_individual.append(np.random.choice(self.characters))
                else:
                    new_individual.append(attribute)
            return np.array(new_individual)

        new_population = []

        for _ in range(self.population_size):
            parents = self.selection_probability()
            child = crossover(parents)
            child = mutate(child)
            new_population.append(child)

        self.population = np.array(new_population)
        self.compute_fitness_score()

    def get_best(self):
        if self.fitness_scores is None:
            return None
        return ''.join(self.population[np.argmax(self.fitness_scores)])
