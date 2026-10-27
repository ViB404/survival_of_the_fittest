from survival_of_the_fittest.genetic_algorithm import GeneticAlgorithm


def main() -> None:
    ga = GeneticAlgorithm(
        num_attributes=2,
        population_size=50
    )

    ga.initialize_population()

    print(ga.population)