import numpy as np
from controller import Supervisor
import random
import math
import csv

# Parâmetros da Simulação
TIME_STEP = 64
POPULATION_SIZE = 10
PARENTS_KEEP = 2
INPUT = 2
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (INPUT * HIDDEN) + HIDDEN + (HIDDEN * OUTPUT) + OUTPUT
GENERATIONS = 3000
EVALUATION_TIME = 200  # segundos
MAX_SPEED = 6.28
BASE_SPEED = 3.0

# --- PARÂMETROS DE MUTAÇÃO ADAPTATIVA ---
# Começa com valores altos para incentivar a exploração
INITIAL_MUTATION_RATE = 0.25
INITIAL_MUTATION_SIZE = 0.5
# Termina com valores baixos para permitir o refinamento
FINAL_MUTATION_RATE = 0.05
FINAL_MUTATION_SIZE = 0.1
# Percentagem de gerações durante a qual a mutação diminui
MUTATION_DECAY_FACTOR = 0.75 
MUTATION_DECAY_GENERATIONS = int(GENERATIONS * MUTATION_DECAY_FACTOR)

def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return (0, 0, 1, angle)

def random_position(min_radius, max_radius, z):
    radius = np.random.uniform(min_radius, max_radius)
    angle = random_orientation()
    x = radius * np.cos(angle[3])
    y = radius * np.sin(angle[3])
    return [x, y, z]

class Evolution:
    def __init__(self):
        self.fitness_history = []
        self.evaluation_start_time = 0
        self.step_count = 0
        
        # Variáveis para a função de fitness moldada
        self.fitness_score = 0.0
        self.last_position = None
        self.found_line = False
        self.time_first_contact = 0.0

        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep() * TIME_STEP)
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        for sensor in self.ground_sensors:
            sensor.enable(self.timestep)

        self.population = [{'genome': np.random.uniform(-1, 1, GENOME_SIZE), 'fitness': 0}
                           for _ in range(POPULATION_SIZE)]

    def reset(self):
        self.robot_node.resetPhysics()
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Zera as variáveis de avaliação
        self.fitness_score = 0.0
        self.step_count = 0
        self.found_line = False
        self.time_first_contact = 0.0

        # Posição e rotação aleatórias
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.rotation_field.setSFRotation(random_rotation)
        pos = random_position(-1, 1, 0.05)
        self.translation_field.setSFVec3f(pos)
        
        self.supervisor.step(0) 
        self.last_position = self.translation_field.getSFVec3f()

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.supervisor.step(self.timestep)

    def select_parents(self):
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        return sorted_population[:PARENTS_KEEP]

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    # A função de mutação agora aceita os parâmetros atuais
    def mutate(self, genome, mutation_rate, mutation_size):
        for i in range(len(genome)):
            if random.random() < mutation_rate:
                genome[i] += np.random.normal(0, mutation_size)
        return genome

    def decode_genome(self, genome):
        idx = 0
        W1 = np.array(genome[idx:idx+INPUT*HIDDEN]).reshape(INPUT, HIDDEN)
        idx += INPUT*HIDDEN
        b1 = np.array(genome[idx:idx+HIDDEN])
        idx += HIDDEN
        W2 = np.array(genome[idx:idx+HIDDEN*OUTPUT]).reshape(HIDDEN, OUTPUT)
        idx += HIDDEN*OUTPUT
        b2 = np.array(genome[idx:idx+OUTPUT])
        return W1, b1, W2, b2

    def run_step(self, genome):
        self.step_count += 1

        is_off_line_left = (self.ground_sensors[0].getValue() / 1023 - 0.6) / 0.2 > 0.3
        is_off_line_right = (self.ground_sensors[1].getValue() / 1023 - 0.6) / 0.2 > 0.3
        is_on_line = not is_off_line_left or not is_off_line_right

        if not self.found_line and is_on_line:
            self.found_line = True
            self.time_first_contact = self.supervisor.getTime() - self.evaluation_start_time

        reward_multiplier = 0.0
        if not is_off_line_left and not is_off_line_right:
            reward_multiplier = 1.0
        elif is_on_line:
            reward_multiplier = 0.5

        sensor_values = [((s.getValue() / 1023.0) - 0.5) * 2 for s in self.ground_sensors]
        W1, b1, W2, b2 = self.decode_genome(genome)
        hidden = np.tanh(np.dot(sensor_values, W1) + b1)
        output = np.tanh(np.dot(hidden, W2) + b2)

        left_speed = BASE_SPEED + abs(output[0]) * (MAX_SPEED - BASE_SPEED)
        right_speed = BASE_SPEED + abs(output[1]) * (MAX_SPEED - BASE_SPEED)

        self.left_motor.setVelocity(max(min(left_speed, MAX_SPEED), 0))
        self.right_motor.setVelocity(max(min(right_speed, MAX_SPEED), 0))

        self.supervisor.step(self.timestep)

        current_position = self.translation_field.getSFVec3f()
        delta_distance = np.linalg.norm(np.array(current_position) - np.array(self.last_position))
        self.last_position = current_position
        self.fitness_score += delta_distance * reward_multiplier

    def run(self):
        start_gen = len(self.fitness_history)

        try:
            for gen in range(start_gen, GENERATIONS):
                print(f"\n=== Geração {gen + 1}/{GENERATIONS} ===")
                
                # --- CÁLCULO DOS PARÂMETROS DE MUTAÇÃO PARA ESTA GERAÇÃO ---
                decay_progress = min(gen / MUTATION_DECAY_GENERATIONS, 1.0)
                current_mutation_rate = INITIAL_MUTATION_RATE - (INITIAL_MUTATION_RATE - FINAL_MUTATION_RATE) * decay_progress
                current_mutation_size = INITIAL_MUTATION_SIZE - (INITIAL_MUTATION_SIZE - FINAL_MUTATION_SIZE) * decay_progress
                print(f"Taxa de Mutação: {current_mutation_rate:.3f}, Tamanho da Mutação: {current_mutation_size:.3f}")

                for idx, individual in enumerate(self.population):
                    self.reset()
                    self.evaluation_start_time = self.supervisor.getTime()

                    print(f"  Indivíduo {idx+1}/{POPULATION_SIZE}", end="", flush=True)

                    while (self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME):
                        self.run_step(individual['genome'])

                    # --- CÁLCULO DA FUNÇÃO DE FITNESS MOLDADA ---
                    fitness_execucao = self.fitness_score
                    fitness_exploracao = 0.0
                    BONUS_MAXIMO = 10.0  # HIPERPARÂMETRO: Ajuste se necessário

                    if self.found_line:
                        time_factor = 1.0 - (self.time_first_contact / EVALUATION_TIME)
                        fitness_exploracao = BONUS_MAXIMO * time_factor

                    fitness = fitness_execucao + fitness_exploracao
                    print(f" - Execução: {fitness_execucao:.2f}, Exploração: {fitness_exploracao:.2f} -> Fitness Total: {fitness:.4f}")
                    individual['fitness'] = fitness

                avg_fitness = sum(ind['fitness'] for ind in self.population) / POPULATION_SIZE
                self.fitness_history.append(avg_fitness)
                print(f"Fitness médio: {avg_fitness:.4f}")

                with open("fitness_history_ANN.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Geração", "FitnessMédio"])
                    for i, fitness in enumerate(self.fitness_history):
                        writer.writerow([i + 1, fitness])

                parents = self.select_parents()
                best_individual = max(self.population, key=lambda ind: ind['fitness'])
                new_population = [best_individual]

                while len(new_population) < POPULATION_SIZE:
                    parent1, parent2 = random.sample(parents, 2)
                    child1_genome, child2_genome = self.crossover(parent1['genome'], parent2['genome'])
                    
                    if len(new_population) < POPULATION_SIZE:
                        # Passa os parâmetros de mutação atuais para a função
                        mutated_genome1 = self.mutate(child1_genome, current_mutation_rate, current_mutation_size)
                        child1 = {'genome': mutated_genome1, 'fitness': 0}
                        new_population.append(child1)
                    if len(new_population) < POPULATION_SIZE:
                        mutated_genome2 = self.mutate(child2_genome, current_mutation_rate, current_mutation_size)
                        child2 = {'genome': mutated_genome2, 'fitness': 0}
                        new_population.append(child2)

                self.population = new_population

        except KeyboardInterrupt:
            print("\nSimulação interrompida!")
            save = input("Deseja salvar a população atual? (s/n): ").lower()
            if save == 's':
                filename = input("Digite o nome do arquivo para salvar (sem extensão): ") + ".npz"
                genomes = [ind['genome'] for ind in self.population]
                fitnesses = [ind['fitness'] for ind in self.population]
                np.savez(filename,
                         genomes=genomes,
                         fitnesses=fitnesses,
                         fitness_history=self.fitness_history)
                print(f"População salva em {filename}")
                print(f"Histórico de fitness salvo em fitness_history_ANN.csv")

def main():
    controller = Evolution()
    controller.run()

if __name__ == "__main__":
    main()
