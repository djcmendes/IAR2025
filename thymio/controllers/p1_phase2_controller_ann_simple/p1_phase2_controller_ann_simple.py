import numpy as np
from controller import Supervisor
import random
import csv


# Genoma inicial pré-definido (melhor comportamento conhecido)
default_genome = np.array([
    # W1 (2x4) - Pesos da camada de entrada para escondida
    1.0, -0.8, 0.5, -0.5,  # Sensor esquerdo
    -0.8, 1.0, -0.5, 0.5,  # Sensor direito
    # b1 (4) - Bias da camada escondida
    0.1, -0.1, 0.2, -0.2,
    # W2 (4x2) - Pesos da camada escondida para saída
    0.8, -0.5,  # Neurônio 1 -> Motores
    -0.5, 0.8,  # Neurônio 2 -> Motores
    0.3, -0.3,  # Neurônio 3 -> Motores
    -0.3, 0.3,  # Neurônio 4 -> Motores
    # b2 (2) - Bias da camada de saída
    0.1, 0.1   # Pequeno bias para manter movimento para frente
], dtype=float)

# Rede neuronal com 2 entradas, 1 camada escondida (4 neurónios), 2 saídas
class SimpleANN:
    def __init__(self, genome):
        i = 0
        self.W1 = genome[i:i+8].reshape((2, 4)); i += 8
        self.b1 = genome[i:i+4];           i += 4
        self.W2 = genome[i:i+8].reshape((4, 2)); i += 8
        self.b2 = genome[i:i+2]

    def forward(self, inputs):
        h = np.tanh(np.dot(inputs, self.W1) + self.b1)
        output = np.tanh(np.dot(h, self.W2) + self.b2)
        return output

class ANNController:
    def __init__(self, supervisor=None, evaluation_time=300):
        # Reuse Supervisor if fornecido, ou crie um
        self.supervisor = supervisor or Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field    = self.robot_node.getField("rotation")
        self.timestep = int(self.supervisor.getBasicTimeStep() * 5)

        # Dispositivos devem vir do Supervisor único
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor= self.supervisor.getDevice('motor.right')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.ground_sensors = [
            self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)
        ]
        for s in self.ground_sensors:
            s.enable(self.timestep)

        self.EVALUATION_TIME = evaluation_time
        self.time_in_line    = 0

    def reset(self):
        self.rotation_field.setSFRotation([0,0,1, np.random.uniform(0,2*np.pi)])
        self.translation_field.setSFVec3f([0,0,0])
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.time_in_line = 0

    def normalize(self, value):
        white, black = 1000.0, 500.0
        return max(0.0, min(1.0, (white - value)/(white - black)))

    def run(self, genome):
        self.reset()
        ann = SimpleANN(genome)
        start_time = self.supervisor.getTime()
        while self.supervisor.getTime() - start_time < self.EVALUATION_TIME:
            s = [self.normalize(sen.getValue()) for sen in self.ground_sensors]
            if s[0] > 0.65 or s[1] > 0.65:
                self.time_in_line += 1
            speeds = ann.forward(np.array(s)) * 6.0
            base = 3.0
            left  = max(0, min(6, base + speeds[0]*2))
            right = max(0, min(6, base + speeds[1]*2))
            self.left_motor.setVelocity(left)
            self.right_motor.setVelocity(right)
            self.supervisor.step(self.timestep)
        return self.time_in_line / self.EVALUATION_TIME

class Evolution:
    def __init__(
        self, controller, pop_size=20, generations=50,
        mutation_rate=0.2, mutation_scale=0.1
    ):
        self.controller   = controller
        self.pop_size     = pop_size
        self.generations  = generations
        self.mut_rate     = mutation_rate
        self.mut_scale    = mutation_scale
        self.genome_len   = 22
        # Inicializa população com o genoma pré-definido em primeiro lugar
        self.population = [default_genome.copy()]
        for _ in range(self.pop_size - 1):
            self.population.append(
                np.random.uniform(-1, 1, self.genome_len)
            )
        self.fitnesses = np.zeros(self.pop_size)

        self.fitness_history = []

    def select_parents(self):
        parents = []
        for _ in range(2):
            i, j = random.sample(range(self.pop_size), 2)
            parents.append(
                self.population[i] if self.fitnesses[i] > self.fitnesses[j]
                else self.population[j]
            )
        return parents

    def crossover(self, p1, p2):
        pt = random.randint(1, self.genome_len - 1)
        return (
            np.concatenate([p1[:pt], p2[pt:]]),
            np.concatenate([p2[:pt], p1[pt:]])
        )

    def mutate(self, genome):
        mask = np.random.rand(self.genome_len) < self.mut_rate
        genome[mask] += np.random.normal(0, self.mut_scale, mask.sum())
        return genome

    def evolve(self):
        with open("fitness_history_ann_simples.csv", mode="w", newline="") as file:  # <<< ADICIONADO
            writer = csv.writer(file)  # <<< ADICIONADO
            writer.writerow(["Generation", "Individual", "Fitness"])  # <<< ADICIONADO

        for gen in range(self.generations):
            print(f"=== Geração {gen+1}/{self.generations} ===")
            for i, genome in enumerate(self.population):
                fit = self.controller.run(genome)
                self.fitnesses[i] = fit
                print(f"Ind {i+1}: fitness={fit:.3f}")
                writer.writerow([gen + 1, i + 1, fit])  # <<< ADICIONADO
                self.fitness_history.append((gen + 1, i + 1, fit))  # <<< ADICIONADO
            idx = np.argsort(-self.fitnesses)
            self.population = [self.population[i] for i in idx]
            self.fitnesses  = self.fitnesses[idx]
            # Elitismo + recombinação
            new_pop = self.population[:2].copy()
            while len(new_pop) < self.pop_size:
                p1, p2 = self.select_parents()
                c1, c2 = self.crossover(p1, p2)
                new_pop += [self.mutate(c1), self.mutate(c2)]
            self.population = new_pop[:self.pop_size]
        best = self.population[0]
        print(f"Melhor genoma fitness={self.fitnesses[0]:.3f}")
        return best

if __name__ == "__main__":
    # Cria um único Supervisor e ANNController
    supervisor = Supervisor()
    controller = ANNController(supervisor=supervisor, evaluation_time=80)

    # Evolução usando o mesmo controller
    evo = Evolution(controller, pop_size=10, generations=30,
                    mutation_rate=0.2, mutation_scale=0.05)
    best_genome = evo.evolve()

    # Avaliação final sem recriar Supervisor
    controller.run(best_genome)
