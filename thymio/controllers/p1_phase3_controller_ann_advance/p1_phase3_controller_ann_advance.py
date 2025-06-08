import numpy as np
from controller import Supervisor
import random
import math
import csv
import os

# Simulation parameters
TIME_STEP = 64
POPULATION_SIZE = 30
PARENTS_KEEP = 4
INPUT = 5 # 2 ground + 3 proximity
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (INPUT * HIDDEN) + HIDDEN + (HIDDEN * OUTPUT) + OUTPUT
GENERATIONS = 300
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 300  # segundos
MAX_SPEED = 6.28
BASE_SPEED = 3.0  # sempre para a frente


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
        
        # NOVAS VARIÁVEIS para fitness baseado em distância
        self.fitness_score = 0.0
        self.last_position = None
        
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
        for sensor in self.ground_sensors:
            sensor.enable(self.timestep)
        self.prox_sensors = [self.__ir_0, self.__ir_2, self.__ir_4] # for the advance

        # Start with a new population
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
        
        # ZERA AS VARIÁVEIS de avaliação para cada indivíduo
        self.fitness_score = 0.0

        self.time_in_line = 0
        self.time_without_collision = 0
        self.time_walking = 0
        self.step_count = 0
        self.collision = False
        self.actual_collision = 0
        self.collision_sensor = 0

        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.rotation_field.setSFRotation(random_rotation)
        pos = random_position(-1, 1, 0.05)
        self.translation_field.setSFVec3f(pos)

        # GARANTE que a posição é atualizada antes de começar
        self.supervisor.step(0)
        self.last_position = self.translation_field.getSFVec3f()

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.supervisor.step(self.timestep)

        self.randomize_obstacles()
        
    def select_parents(self):
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        return sorted_population[:PARENTS_KEEP]

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
        W1 = np.array(genome[idx:idx + INPUT * HIDDEN]).reshape(INPUT, HIDDEN)
        idx += INPUT * HIDDEN
        b1 = np.array(genome[idx:idx + HIDDEN])
        idx += HIDDEN
        W2 = np.array(genome[idx:idx + HIDDEN * OUTPUT]).reshape(HIDDEN, OUTPUT)
        idx += HIDDEN * OUTPUT
        b2 = np.array(genome[idx:idx + OUTPUT])
        return W1, b1, W2, b2

    def run_step(self, genome):
        self.step_count += 1

        # Sensor collision detection disabled (not used here)

        self.collision = bool( # Just for monitoring
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

        # 1. Verifica se o robô está na linha (assumindo linha branca/clara)
        # A lógica `> 0.3` provavelmente detecta o chão escuro (fora da linha)
        is_off_line_left = (self.ground_sensors[0].getValue() / 1023 - 0.6) / 0.2 > 0.3
        is_off_line_right = (self.ground_sensors[1].getValue() / 1023 - 0.6) / 0.2 > 0.3

        reward_multiplier = 0.0
        if not is_off_line_left and not is_off_line_right:
            reward_multiplier = 50.0  # Recompensa máxima por estar centrado
        elif not is_off_line_left or not is_off_line_right:
            reward_multiplier = 40.0  # Recompensa parcial para incentivar a correção

        # 2. Roda a rede neural para obter a velocidade dos motores
        ground_sensor_values = [((s.getValue() / 1023.0) - 0.5) * 2 for s in self.ground_sensors]

        prox_sensor_values = [p.getValue() / 4300 for p in self.prox_sensors]

        # Considera colisão se algum sensor de proximidade estiver muito ativo
        proximity_sensor_values = [p.getValue() for p in self.prox_sensors]
        # Se nenhum sensor está acima de 4000
        #average_speed = (left_speed + right_speed) / 2

        if any(value > 4000 for value in proximity_sensor_values):
            self.collision_sensor += 1
            self.time_without_collision += -10
            #average_speed = 0
        elif not any(value > 3000 for value in proximity_sensor_values):
            self.time_without_collision += 2.0
        elif not any((4000 < value < 3000) for value in proximity_sensor_values): # Se nenhum sensor está acima de 4300
            self.time_without_collision += 1.0

        # Decode genome into neural network weights
        W1, b1, W2, b2 = self.decode_genome(genome)
        hidden = np.tanh(np.dot(ground_sensor_values + prox_sensor_values, W1) + b1)
        output = np.tanh(np.dot(hidden, W2) + b2)

        left_speed = BASE_SPEED + output[0] * (MAX_SPEED - BASE_SPEED)
        right_speed = BASE_SPEED + output[1] * (MAX_SPEED - BASE_SPEED)

        #if not average_speed < 0.01:  # Threshold para considerar "parado" ou "marcha-atrás"
        #    self.time_walking += 1
        #else:
        #    self.time_walking -= 5

        self.left_motor.setVelocity(max(min(left_speed, MAX_SPEED), 0))
        self.right_motor.setVelocity(max(min(right_speed, MAX_SPEED), 0))

        # 3. Avança a simulação
        self.supervisor.step(self.timestep)

        # 4. Calcula a distância percorrida neste passo
        current_position = self.translation_field.getSFVec3f()
        delta_distance = np.linalg.norm(np.array(current_position) - np.array(self.last_position))
        self.last_position = current_position

        # 5. Adiciona à pontuação de fitness (distância * recompensa por estar na linha)
        self.fitness_score += delta_distance * reward_multiplier

    def run(self):
        start_gen = len(self.fitness_history)

        try:
            for gen in range(start_gen, GENERATIONS):
                print(f"\n=== Generation {gen + 1}/{GENERATIONS} ===")

                for idx, individual in enumerate(self.population):
                    self.reset()
                    self.evaluation_start_time = self.supervisor.getTime()
                    total_steps = 0
                    # Run simulation for evaluation period

                    while (self.supervisor.getTime() - self.evaluation_start_time) < EVALUATION_TIME:
                        self.run_step(individual['genome'])
                        total_steps += 1                   

                    collisions_fitness = (self.time_without_collision / EVALUATION_TIME)
                    
                    # A pontuação de fitness agora é a distância total percorrida na linha
                    fitness = self.fitness_score + collisions_fitness
                    #print(f" - Fitness: {fitness:.4f}", flush=True)

                    #score_collision    = self.time_without_collision / total_steps
                    #score_time_in_line = self.time_in_line / total_steps
                    #score_time_walking = self.time_walking / total_steps
                    #if score_time_in_line > 0.95: # Discard invalid fitness values
                    #    fitness = 0
                    #elif score_time_in_line < 0.1:
                    #    fitness = (0.0 * score_collision + 1.0 * score_time_in_line + score_time_walking * 0.0)
                    #else:
                    #    fitness = (0.25 * score_collision + 0.7 * score_time_in_line + 0.05 * score_time_walking)

                    #fitness = min(1.0, max(0.0, fitness))

                    individual['fitness'] = fitness
                    """
                    individual['score_time_in_line'] = score_time_in_line
                    individual['score_collision'] = score_collision
                    individual['score_time_walking'] = score_time_walking
                    """
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

                """
                print(f"\n=== Averages for Generation {gen + 1} ===")
                avg = sum(ind['score_time_in_line'] for ind in self.population) / POPULATION_SIZE
                self.score_time_in_line_history.append(avg)
                print(f"Average score_time_in_line: {avg:.4f}")
                

                avg = sum(ind['score_collision'] for ind in self.population) / POPULATION_SIZE
                self.score_collision_history.append(avg)
                print(f"Average score_collision: {avg:.4f}") # only for 3 proximity sensores

                """

                avg = sum(ind['fitness'] for ind in self.population) / POPULATION_SIZE
                self.fitness_history.append(avg)
                print(f"Average fitness: {avg:.4f}")

                """
                avg = sum(ind['score_time_walking'] for ind in self.population) / POPULATION_SIZE
                self.score_time_walking_history.append(avg)
                print(f"Average score_time_walking: {avg:.4f}")
                
                # Calculate and store average fitness
                avg = sum(ind['collision_sensor'] for ind in self.population) / POPULATION_SIZE
                self.collision_sensor_history.append(avg)
                print(f"Average collisions sensor (Detected): {avg}") # actual means check every sensor for collision
                """
                # Calculate and store average fitness
                avg = sum(ind['collisions'] for ind in self.population) / POPULATION_SIZE
                self.collisions_history.append(avg)
                print(f"Average collisions (actual): {avg}") # actual means check every sensor for collision

                self.genome_history.append([ind['genome'] for ind in self.population])

                # Save fitness history to CSV
                with open("../../../data/fitness_history_ANN_advanced_4n.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Generation",
                        "AverageFitness",
                        #"AverageTimeInLineScore",
                        #"AverageCollisionScore",
                        #"AverageTimeWalkingScore",
                        #"AverageDetectedCollisions",
                        "AverageActualCollisions",
                        "genome"
                    ])
                    for i, fitness in enumerate(self.fitness_history):
                        writer.writerow([
                            i + 1,
                            fitness,
                            #self.score_time_in_line_history[i],
                            #self.score_collision_history[i],
                            #self.score_time_walking_history[i],
                            #self.collision_sensor_history[i],
                            self.collisions_history[i],
                            self.genome_history[i]
                        ])

                # Evolutionary algorithm
                parents = self.select_parents()
                new_population = parents.copy()

                # Elitismo: o melhor indivíduo vai diretamente para a próxima geração
                best_individual = max(self.population, key=lambda ind: ind['fitness'])
                new_population = [best_individual]

                # Preenche o resto da população com filhos
                while len(new_population) < POPULATION_SIZE:
                    parent1, parent2 = random.sample(parents, 2)
                    child1_genome, child2_genome = self.crossover(parent1['genome'], parent2['genome'])

                    # Garante que não exceda o tamanho da população se for ímpar
                    if len(new_population) < POPULATION_SIZE:
                        child1 = {'genome': self.mutate(child1_genome), 'fitness': 0}
                        new_population.append(child1)
                    if len(new_population) < POPULATION_SIZE:
                        child2 = {'genome': self.mutate(child2_genome), 'fitness': 0}
                        new_population.append(child2)

                self.population = new_population

        except KeyboardInterrupt:
            print("\nSimulation interrupted!")
            save = input("Do you want to save the current population? (y/n): ").lower()
            if save == 'y':
                filename = input("Enter filename to save (without extension): ") + ".npz"

                genomes = [ind['genome'] for ind in self.population]
                fitnesses = [ind['fitness'] for ind in self.population]
                #time_in_line_scores = [ind['score_time_in_line'] for ind in self.population]
                #collision_scores = [ind['score_collision'] for ind in self.population]
                #time_walking_scores = [ind['score_time_walking'] for ind in self.population]
                collisions_sensor = [ind['collision_sensor'] for ind in self.population]
                collisions = [ind['collisions'] for ind in self.population]

                np.savez(
                    filename,
                    genomes=genomes,
                    fitnesses=fitnesses,
                    fitness_history = self.fitness_history,
                    #time_in_line_scores=time_in_line_scores,
                    #collision_scores=collision_scores,
                    #time_in_walking_scores=time_walking_scores,
                    #collisions_sensor=collisions_sensor,
                    collisions=collisions,

                    #score_time_in_line_history = self.score_time_in_line_history,
                    #score_collision_history = self.score_collision_history,
                    #score_time_walking=self.score_time_walking_history,
                    #collisions_sensor_history = self.collision_sensor_history,
                    collisions_history = self.collisions_history,
                )

                print(f"Population saved to {filename}")
                print(f"Fitness history saved to fitness_history_ANN_4n.csv")

def main():
    controller = Evolution()
    controller.run()
    exit()

if __name__ == "__main__":
    main()