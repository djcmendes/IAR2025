import numpy as np
import random
from ann_controller import ANNController

POPULATION_SIZE = 10
GENOME_SIZE = 22
GENERATIONS = 50
PARENTS_KEEP = 3
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.1


def mutate(genome):
    new_genome = genome.copy()
    for i in range(len(new_genome)):
        if random.random() < MUTATION_RATE:
            new_genome[i] += np.random.normal(0, MUTATION_SIZE)
    return new_genome


def crossover(parent1, parent2):
    alpha = np.random.uniform(0, 1, size=len(parent1))
    child = alpha * parent1 + (1 - alpha) * parent2
    return child


def evaluate_genome(controller, genome):
    controller.run(genome)
    
    # Critério de fitness básico: distância percorrida no eixo x
    pos = controller.robot.getPosition()
    return pos[0]  # pode adaptar para total da trajetória


def main():
    controller = ANNController()
    population = [np.random.uniform(-1, 1, GENOME_SIZE) for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        print(f"Geração {generation}")

        fitnesses = []
        for i, genome in enumerate(population):
            print(f"  Avaliando indivíduo {i}...")
            controller.reset_position()
            fitness = evaluate_genome(controller, genome)
            fitnesses.append((fitness, genome))

        # Selecionar os melhores
        fitnesses.sort(reverse=True, key=lambda x: x[0])
        print(f"  Melhor fitness: {fitnesses[0][0]:.2f}")
        new_population = [fit[1] for fit in fitnesses[:PARENTS_KEEP]]

        # Gerar novos filhos até completar a população
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(new_population, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population


if __name__ == "__main__":
    main()
