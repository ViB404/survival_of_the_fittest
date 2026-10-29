import numpy as np

from survival_of_the_fittest import fitness
from survival_of_the_fittest.genetic_algorithm import GeneticAlgorithm


def main() -> None:
    target_sentence = 'Hello World'
    MAX_GEN = 1000
    MAX_SUCCESS = 3
    NUM_ATTRIBUTES = len(target_sentence)
    POPULATION_SIZE = 500
    MUTATION_PROB = .01
    CROSSOVER_PROB = .75

    GA = GeneticAlgorithm(fitness_function=fitness.fitness_function,
                          num_attributes=NUM_ATTRIBUTES,
                          population_size=POPULATION_SIZE,
                          mut_prob=MUTATION_PROB,
                          cross_prob=CROSSOVER_PROB)

    GA.initialize_population()

    scores = []
    generation_counter = 0
    success_counter = 0

    while generation_counter <= MAX_GEN and success_counter < MAX_SUCCESS:
        GA.compute_fitness_score()
        scores.append(np.mean(GA.fitness_scores))

        print(f"Gen {generation_counter:3d} | Avg Score: {scores[generation_counter]:8.4f} | Best: {GA.get_best()}")

        if GA.get_best() == target_sentence:
            success_counter += 1

        GA.run()

        generation_counter += 1


if __name__ == "__main__":
    main()
