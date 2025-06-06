import numpy as np
from controller import Supervisor
import random
import math
import csv
import os

# Simulation parameters
TIME_STEP = 64
POPULATION_SIZE = 10
PARENTS_KEEP = 3
INPUT = 5 # 2 ground + 3 proximity
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (INPUT * HIDDEN) + HIDDEN + (HIDDEN * OUTPUT) + OUTPUT
GENERATIONS = 300
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 300  # Simulated seconds per individual
MAX_SPEED = 6.28
BASE_SPEED = 1.0  # Minimal always-forward motion

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
        self.score_time_in_line_history = []
        self.score_collision_history = []
        self.score_time_walking_history = []
        self.collision_sensor_history = []
        self.collisions_history = []
        self.genome_history = []

        self.evaluation_start_time = 0
        self.collision = False
        self.step_count = 0
        self.time_in_line = 0
        self.time_walking = 0
        self.time_without_collision = 0
        self.collision_sensor = 0 # collision captured by the 3 proximity sensores
        self.actual_collision = 0

        # Supervisor to reset robot position
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
   
        self.robot_node = self.supervisor.getFromDef("ROBOT") 
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep() * TIME_STEP)
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')

        self.__ir_0 = self.supervisor.getDevice('prox.horizontal.0')
        self.__ir_1 = self.supervisor.getDevice('prox.horizontal.1')
        self.__ir_2 = self.supervisor.getDevice('prox.horizontal.2')
        self.__ir_3 = self.supervisor.getDevice('prox.horizontal.3')
        self.__ir_4 = self.supervisor.getDevice('prox.horizontal.4')
        self.__ir_5 = self.supervisor.getDevice('prox.horizontal.5')
        self.__ir_6 = self.supervisor.getDevice('prox.horizontal.6')
        self.__ir_7 = self.supervisor.getDevice('prox.ground.0')
        self.__ir_8 = self.supervisor.getDevice('prox.ground.1')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.__ir_0.enable(self.timestep)
        self.__ir_1.enable(self.timestep)
        self.__ir_2.enable(self.timestep)
        self.__ir_3.enable(self.timestep)
        self.__ir_4.enable(self.timestep)
        self.__ir_5.enable(self.timestep)
        self.__ir_6.enable(self.timestep)
        self.__ir_7.enable(self.timestep)
        self.__ir_8.enable(self.timestep)

        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        self.prox_sensors = [self.__ir_0, self.__ir_2, self.__ir_4] # for the advance

        # Ask user for training mode
        # mode = input("Start new training (n) or load from file (l)? ").lower()
        # if mode == 'l':
        #     filename = input("Enter filename to load (without extension): ") + ".npz"
        #     try:
        #         data = np.load(filename)
        #         genomes = data['genomes']
        #         fitnesses = data['fitnesses']
        #         self.fitness_history = list(data['fitness_history'])
        #         self.population = [{'genome': genome, 'fitness': fitness} 
        #                           for genome, fitness in zip(genomes, fitnesses)]
        #         print(f"Loaded population from {filename}")
        #         print(f"Resuming from generation {len(self.fitness_history)}")
        #     except Exception as e:
        #         print(f"Error loading file: {e}. Starting new population.")
        #         self.population = [{'genome': np.random.uniform(-1, 1, GENOME_SIZE), 'fitness': 0}
        #                    for _ in range(POPULATION_SIZE)]
        # else:
        self.population = [{'genome': np.random.uniform(-1, 1, GENOME_SIZE), 'fitness': 0}
                    for _ in range(POPULATION_SIZE)]

    def randomize_obstacles(self):
        # Aceder ao campo children da raiz da cena
        root = self.supervisor.getRoot()
        children_field = root.getField("children")
        # Eliminar caixas anteriores (limpeza opcional)
        for i in reversed(range(children_field.getCount())):
            node = children_field.getMFNode(i)
            if hasattr(node, 'getDef') and node.getDef() and node.getDef().startswith("WHITE_BOX_"):
                children_field.removeMF(i)

        # Gerar novos obstáculos
        def random_orientation():
            angle = np.random.uniform(0, 2 * np.pi)
            return (0, 1, 0, angle)  # rotação no eixo Y

        def random_position(exclusion_radius=0.2):
            while True:
                x = np.random.uniform(-1.5, 1.5)
                y = np.random.uniform(-1.5, 1.5)

                # Evita gerar no centro
                if np.sqrt(x**2 + y**2) > exclusion_radius:
                    return x, y, 1.01  # z fixo (altura do obstáculo)

        for i in range(5):
            position = random_position()
            orientation = random_orientation()
            length = np.random.uniform(0.05, 0.2)
            width = np.random.uniform(0.05, 0.2)

            box_string = f"""
            DEF WHITE_BOX_{i} Solid {{
            translation {position[0]} {position[1]} {position[2]}
            rotation {orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]}
            physics Physics {{
                density 1000.0
            }}
            children [
                Shape {{
                appearance Appearance {{
                    material Material {{
                    diffuseColor 1 1 1
                    }}
                }}
                geometry Box {{
                    size {length} {width} 0.2
                }}
                }}
            ]
            boundingObject Box {{
                size {length} {width} 0.2
            }}
            }}
            """

            children_field.importMFNodeFromString(-1, box_string)

    def reset(self):
        self.robot_node.resetPhysics()
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.time_in_line = 0
        self.time_without_collision = 0
        self.time_walking = 0
        self.step_count = 0
        self.collision = False
        self.actual_collision = 0
        self.collision_sensor = 0

        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.robot_node.getField('rotation').setSFRotation(random_rotation)
        pos = random_position(-1, 1, 0.05)
        self.robot_node.getField('translation').setSFVec3f(pos)
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.supervisor.step(self.timestep)

        self.randomize_obstacles()
        
    def select_parents(self):
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        parents = sorted_population[:PARENTS_KEEP]
        return parents

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(self, genome):
        for i in range(len(genome)):
            if random.random() < MUTATION_RATE:
                genome[i] += np.random.normal(0, MUTATION_SIZE)
        return genome

    def decode_genome(self, genome):
        idx = 0
        
        # Input to hidden weights (2x4)
        W1 = np.array(genome[idx:idx+INPUT*HIDDEN]).reshape(INPUT, HIDDEN)
        idx += INPUT*HIDDEN
        
        # Hidden biases (4)
        b1 = np.array(genome[idx:idx+HIDDEN])
        idx += HIDDEN
        
        # Hidden to output weights (4x2)
        W2 = np.array(genome[idx:idx+HIDDEN*OUTPUT]).reshape(HIDDEN, OUTPUT)
        idx += HIDDEN*OUTPUT
        
        # Output biases (2)
        b2 = np.array(genome[idx:idx+OUTPUT])
        
        return W1, b1, W2, b2

    def run_step(self, genome):
        self.step_count += 1
        
        # Check for collisions (monitor, not for learning)
        self.collision = bool(
            self.step_count > 10 and
            (self.__ir_0.getValue() > 4300 or 
             self.__ir_1.getValue() > 4300 or
             self.__ir_2.getValue() > 4300 or
             self.__ir_3.getValue() > 4300 or
             self.__ir_4.getValue() > 4300 or
             self.__ir_5.getValue() > 4300 or
             self.__ir_6.getValue() > 4300)
        )

        if self.collision:
            self.actual_collision += 1

        # Read and normalize ground sensors
        ground_sensor_values = [s.getValue() / 1023.0 for s in self.ground_sensors]
        prox_sensor_values = [p.getValue() / 4300 for p in self.prox_sensors]

        # Decode genome into neural network weights
        W1, b1, W2, b2 = self.decode_genome(genome)

        hidden = np.tanh(np.dot(ground_sensor_values + prox_sensor_values, W1) + b1)
        output = np.tanh(np.dot(hidden, W2) + b2)

        left_speed = output[0] * MAX_SPEED
        right_speed = output[1] * MAX_SPEED

        self.left_motor.setVelocity(max(left_speed, 0))
        self.right_motor.setVelocity(max(right_speed, 0))

        ###### -- metrics for fitness

        ground_sensor_left = (self.ground_sensors[0].getValue() / 1023 - .6) / .2 > .3  # Linha Preta
        ground_sensor_right = (self.ground_sensors[1].getValue() / 1023 - .6) / .2 > .3

        # Corrected condition order
        if not ground_sensor_left and not ground_sensor_right:
            self.time_in_line += 1  # Higher reward for centered
        elif not ground_sensor_left or not ground_sensor_right:
            self.time_in_line += 0.5  # Lower reward for partial contact

        # Considera colisão se algum sensor de proximidade estiver muito ativo
        proximity_sensor_values = [p.getValue() for p in self.prox_sensors]
        # Se nenhum sensor está acima de 4000
        average_speed = (left_speed + right_speed) / 2

        if any(value > 4000 for value in proximity_sensor_values):
            self.collision_sensor += 1
            self.time_without_collision += -10
            average_speed = 0
        elif not any(value > 3000 for value in proximity_sensor_values):
            self.time_without_collision += 1
        elif not any((4000 < value < 3000) for value in proximity_sensor_values): # Se nenhum sensor está acima de 4300
            self.time_without_collision += 0.5

        if not average_speed < 0.01:  # Threshold para considerar "parado" ou "marcha-atrás"
            self.time_walking += 1
        else:
            self.time_walking -= 5

        ########

        self.supervisor.step(self.timestep)

    def run(self):
        start_gen = len(self.fitness_history)

        try:
            for gen in range(start_gen, GENERATIONS):
                print(f"\n=== Generation {gen + 1}/{GENERATIONS} ===")

                # Evaluate each individual
                for idx, individual in enumerate(self.population):
                    self.reset()
                    self.evaluation_start_time = self.supervisor.getTime()
                    total_steps = 0
                    # Run simulation for evaluation period

                    while (self.supervisor.getTime() - self.evaluation_start_time) < EVALUATION_TIME:
                        self.run_step(individual['genome'])
                        total_steps += 1

                    score_collision    = self.time_without_collision / total_steps
                    score_time_in_line = self.time_in_line / total_steps
                    score_time_walking = self.time_walking / total_steps

                    if score_time_in_line > 0.95: # Discard invalid fitness values
                        fitness = 0
                    elif score_time_in_line < 0.1:
                        fitness = (0.0 * score_collision + 1.0 * score_time_in_line + score_time_walking * 0.0)
                    else:
                        fitness = (0.25 * score_collision + 0.7 * score_time_in_line + 0.05 * score_time_walking)

                    fitness = min(1.0, max(0.0, fitness))

                    individual['fitness'] = fitness
                    individual['score_time_in_line'] = score_time_in_line
                    individual['score_collision'] = score_collision
                    individual['score_time_walking'] = score_time_walking
                    individual['collision_sensor'] = self.collision_sensor
                    individual['collisions'] = self.actual_collision

                    """
                    print(f"GEN {gen + 1} - Individual {idx + 1}/{POPULATION_SIZE}", flush=True)
                    print(f"fitness: {fitness:.2f}", flush=True)
                    print(f"score_time_in_line: {score_time_in_line:.2f}", flush=True)
                    print(f"score_collision: {score_collision:.2f}", flush=True)
                    print(f"score_time_walking: {score_time_walking:.2f}", flush=True)
                    print(f"detected_collisions: {self.collision_sensor:.2f}", flush=True)
                    print(f"collisions: {self.actual_collision:.2f}", flush=True)
                    """


                print(f"\n=== Averages for Generation {gen + 1} ===")
                avg = sum(ind['score_time_in_line'] for ind in self.population) / POPULATION_SIZE
                self.score_time_in_line_history.append(avg)
                print(f"Average score_time_in_line: {avg:.4f}")

                avg = sum(ind['score_collision'] for ind in self.population) / POPULATION_SIZE
                self.score_collision_history.append(avg)
                print(f"Average score_collision: {avg:.4f}") # only for 3 proximity sensores

                avg = sum(ind['fitness'] for ind in self.population) / POPULATION_SIZE
                self.fitness_history.append(avg)
                print(f"Average fitness: {avg:.4f}")

                avg = sum(ind['score_time_walking'] for ind in self.population) / POPULATION_SIZE
                self.score_time_walking_history.append(avg)
                print(f"Average score_time_walking: {avg:.4f}")

                # Calculate and store average fitness
                avg = sum(ind['collision_sensor'] for ind in self.population) / POPULATION_SIZE
                self.collision_sensor_history.append(avg)
                print(f"Average collisions sensor (Detected): {avg}") # actual means check every sensor for collision

                # Calculate and store average fitness
                avg = sum(ind['collisions'] for ind in self.population) / POPULATION_SIZE
                self.collisions_history.append(avg)
                print(f"Average collisions (actual): {avg}") # actual means check every sensor for collision

                self.genome_history.append([ind['genome'] for ind in self.population])

                # Save fitness history to CSV
                with open("../../../data/fitness_history_ANN_advanced.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Generation",
                        "AverageFitness",
                        "AverageTimeInLineScore",
                        "AverageCollisionScore",
                        "AverageTimeWalkingScore",
                        "AverageDetectedCollisions",
                        "AverageActualCollisions",
                        "genome"
                    ])
                    for i, fitness in enumerate(self.fitness_history):
                        writer.writerow([
                            i + 1,
                            fitness,
                            self.score_time_in_line_history[i],
                            self.score_collision_history[i],
                            self.score_time_walking_history[i],
                            self.collision_sensor_history[i],
                            self.collisions_history[i],
                            self.genome_history[i]
                        ])

                # Evolutionary algorithm
                parents = self.select_parents()
                new_population = parents.copy()

                # elitimo keep 1 best
                ind_validos = [ind for ind in self.population if ind['fitness'] > 0.2]
                # Se houver algum, pega o melhor entre eles
                best_individual = None
                if ind_validos:
                    best_individual = max(ind_validos, key=lambda ind: ind['fitness'])

                # Breed new population
                while len(new_population) < POPULATION_SIZE:
                    parent1, parent2 = random.sample(parents, 2)
                    child1_genome, child2_genome = self.crossover(
                        parent1['genome'], parent2['genome'])

                    child1 = {
                        'genome': self.mutate(child1_genome),
                        'fitness': 0
                    }
                    child2 = {
                        'genome': self.mutate(child2_genome),
                        'fitness': 0
                    }
                    new_population.extend([child1, child2])

                if best_individual:
                    self.population = [best_individual] + new_population[:POPULATION_SIZE - 1]
                else:
                    self.population = new_population[:POPULATION_SIZE]


        except KeyboardInterrupt:
            print("\nSimulation interrupted!")
            save = input("Do you want to save the current population? (y/n): ").lower()
            if save == 'y':
                filename = input("Enter filename to save (without extension): ") + ".npz"

                genomes = [ind['genome'] for ind in self.population]
                fitnesses = [ind['fitness'] for ind in self.population]
                time_in_line_scores = [ind['score_time_in_line'] for ind in self.population]
                collision_scores = [ind['score_collision'] for ind in self.population]
                time_walking_scores = [ind['score_time_walking'] for ind in self.population]
                collisions_sensor = [ind['collision_sensor'] for ind in self.population]
                collisions = [ind['collisions'] for ind in self.population]

                np.savez(
                    filename,
                    genomes=genomes,
                    fitnesses=fitnesses,
                    time_in_line_scores=time_in_line_scores,
                    collision_scores=collision_scores,
                    time_in_walking_scores=time_walking_scores,
                    collisions_sensor=collisions_sensor,
                    collisions=collisions,
                    fitness_history = self.fitness_history,
                    score_time_in_line_history = self.score_time_in_line_history,
                    score_collision_history = self.score_collision_history,
                    score_time_walking=self.score_time_walking_history,
                    collisions_sensor_history = self.collision_sensor_history,
                    collisions_history = self.collisions_history,
                )

                print(f"Population saved to {filename}")
                print(f"Fitness history saved to fitness_history_ANN.csv")

                # Display fitness graph data
                print("\nFitness history:")
                for gen, fitness in enumerate(self.fitness_history):
                    print(
                        f" Generation {gen+1}: {fitness:.4f}",
                        f", {self.score_time_in_line_history[gen]:.4f}",
                        f", {self.score_collision_history[gen]:.4f}",
                        f", {self.score_time_walking_history[gen]:.4f}",
                        f", {self.collision_sensor_history[gen]:.4f}",
                        f", {self.collisions_history[gen]:.4f}"
                    )
            return

def main():
    controller = Evolution()
    controller.run()
    exit()

if __name__ == "__main__":
    main()