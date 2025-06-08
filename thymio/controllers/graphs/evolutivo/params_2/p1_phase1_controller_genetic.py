import numpy as np
from controller import Supervisor
import random
import math
import csv

# Simulation parameters
TIME_STEP = 64
POPULATION_SIZE = 10
PARENTS_KEEP = 3
INPUT = 5
OUTPUT = 2
GENERATIONS = 3000
MUTATION_RATE = 0.3
MUTATION_SIZE = 0.1
EVALUATION_TIME = 300  # Simulated seconds per individual
RANGE = 5


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
        self.collision = False

        # Supervisor to reset robot position
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
   
        self.robot_node = self.supervisor.getFromDef("ROBOT") 
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep()*TIME_STEP)
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

        self.sensors = [self.__ir_0,self.__ir_2,self.__ir_4]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]

        self.__n = 0
        self.prev_position = self.supervisor.getSelf().getPosition()
        self.time_in_line = 0
        
        self.population = [{'weights': np.random.uniform(0, 1, 6), 'fitness': 0}
                   for _ in range(POPULATION_SIZE)]
        

        

    def reset(self, seed=None, options=None):
        self.time_in_line = 0
        # self.__n = 0
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.supervisor.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)
        pos = random_position(-1, 1, 0.1)
        # print(f"Random position: {pos}")
        self.supervisor.getFromDef('ROBOT').getField('translation').setSFVec3f(pos)
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        
    def select_parents(self):
        # Ordena pela fitness (maior é melhor)
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        parents = sorted_population[:PARENTS_KEEP]
        return parents

    def crossover(self, parent1, parent2):
        # Crossover de um ponto
        point = random.randint(1, len(parent1) - 1)
        child1_weights = np.concatenate([parent1[:point], parent2[point:]])
        child2_weights = np.concatenate([parent2[:point], parent1[point:]])
        return child1_weights, child2_weights

    def mutate(self, weights):
        for i in range(len(weights)):
            if random.random() < MUTATION_RATE:
                weights[i] += np.random.normal(0, MUTATION_SIZE)
        return weights

    def runStep(self, weights):
        # self.__n += 1  
        self.collision = bool(
                self.__n > 10 and
                (self.__ir_0.getValue()>4300 or 
                self.__ir_1.getValue()>4300 or
                self.__ir_2.getValue()>4300 or
                self.__ir_3.getValue()>4300 or
                self.__ir_4.getValue()>4300 or
                self.__ir_5.getValue()>4300 or
                self.__ir_6.getValue()>4300)
            )
        
        ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2>.3 # True -> chao Flase -> Linha Preta
        ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2>.3
        # print(f"Ground sensor left: {ground_sensor_left}, Ground sensor right: {ground_sensor_right}")
        # if ground_sensor_left or ground_sensor_right:
        #     self.time_in_line += 1

        if not ground_sensor_left or not ground_sensor_right:
            self.time_in_line += 5
        elif not ground_sensor_left and not ground_sensor_right:
            self.time_in_line += 10
        # return self.time_in_line

        left_speed =  ground_sensor_left * weights[0] + ground_sensor_right * weights[1] + weights[2]
        right_speed = ground_sensor_left * weights[3] + ground_sensor_right * weights[4] + weights[5]
        
        self.left_motor.setVelocity(max(min(left_speed, 9), -9)) # Cap na velocidade
        self.right_motor.setVelocity(max(min(right_speed, 9), -9))

        self.supervisor.step(self.timestep)

        # self.calculateFitness(ground_sensor_left,ground_sensor_right)
        
   
    def run(self):
        for g in range(GENERATIONS):
            print(f"\n=== Geração {g + 1} ===")
            k = 0
            for i in self.population:
                k += 1
                self.evaluation_start_time = self.supervisor.getTime()
                # print(f"Indivíduo: {k} {i['weights']}")
                self.reset()
                while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
                    self.runStep(i['weights'])
                # print(f"Fitness: {i['fitness']}")
                i['fitness'] = self.time_in_line / EVALUATION_TIME
                print(f"Indivíduo: {k} {i['weights']} Fitness: {i['fitness']}")
            parents = self.select_parents()
            average_fitness = sum(i['fitness'] for i in self.population) / POPULATION_SIZE
            self.fitness_history.append(average_fitness)

            # Nova geração com pais + filhos
            new_population = parents.copy()
            while len(new_population) < POPULATION_SIZE:
                p1, p2 = random.sample(parents, 2)
                c1_weights, c2_weights = self.crossover(p1['weights'], p2['weights'])
                c1 = {'weights': self.mutate(c1_weights)}
                c2 = {'weights': self.mutate(c2_weights)}
                new_population.extend([c1, c2])

            self.population = new_population[:POPULATION_SIZE]

            with open("fitness_history_Braitenberg.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Generation", "AverageFitness"])
                for i, fitness in enumerate(self.fitness_history):
                    writer.writerow([i + 1, fitness])


# Main evolutionary loop
def main():
    # Run the evolutionary algorithm
    controller = Evolution()
    controller.run()

if __name__ == "__main__":
    main() 