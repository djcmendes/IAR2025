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
INPUT = 2
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (INPUT * HIDDEN) + HIDDEN + (HIDDEN * OUTPUT) + OUTPUT
GENERATIONS = 300
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 300  # Simulated seconds per individual
MAX_SPEED = 9.0
BASE_SPEED = 1.0  # Minimal always-forward motion
SENSOR_THRESHOLD = 0.66  # Threshold for detecting black line

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
        self.step_count = 0
        self.time_in_line = 0

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
        
    def reset(self):
        self.time_in_line = 0
        self.step_count = 0
        self.collision = False
        
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.robot_node.getField('rotation').setSFRotation(random_rotation)
        pos = random_position(-1, 1, 0.1)
        self.robot_node.getField('translation').setSFVec3f(pos)
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.supervisor.step(self.timestep)
        
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
        
        # Check for collisions
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
        
        # Read and normalize ground sensors
        sensor_values = [s.getValue() / 1023.0 for s in self.ground_sensors]
        
        # Calculate fitness
        # sensor_left = sensor_values[0] < SENSOR_THRESHOLD
        # sensor_right = sensor_values[1] < SENSOR_THRESHOLD

        ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2>.3 # True -> chao Flase -> Linha Preta
        ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2>.3
        
        if not ground_sensor_left or not ground_sensor_right:
            self.time_in_line += 5
        elif not ground_sensor_left and not ground_sensor_right:
            self.time_in_line += 10
        
        # Decode genome into neural network weights
        W1, b1, W2, b2 = self.decode_genome(genome)
        
        # Neural network forward pass
        hidden = np.tanh(np.dot(sensor_values, W1) + b1)
        output = np.tanh(np.dot(hidden, W2) + b2)
        
        # MODIFICATION: Ensure minimal forward motion
        # Set motor speeds (base forward + network modulation)
        left_speed = BASE_SPEED + output[0] * (MAX_SPEED - BASE_SPEED)
        right_speed = BASE_SPEED + output[1] * (MAX_SPEED - BASE_SPEED)
        
        self.left_motor.setVelocity(max(min(left_speed, MAX_SPEED), 0))
        self.right_motor.setVelocity(max(min(right_speed, MAX_SPEED), 0))
        
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
                    
                    print(f"  Individual {idx+1}/{POPULATION_SIZE}", end="", flush=True)
                    
                    # Run simulation for evaluation period
                    while (self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME 
                           and not self.collision):
                        self.run_step(individual['genome'])
                    
                    # MODIFICATION: Discard invalid fitness values
                    fitness = self.time_in_line / EVALUATION_TIME
                    if fitness > 10.0:
                        print(f" - Invalid fitness {fitness:.2f} > 10 - setting to 0")
                        fitness = 0.0
                    else:
                        print(f" - Fitness: {fitness:.4f}")
                    
                    individual['fitness'] = fitness
                
                # Calculate and store average fitness
                avg_fitness = sum(ind['fitness'] for ind in self.population) / POPULATION_SIZE
                self.fitness_history.append(avg_fitness)
                print(f"Average fitness: {avg_fitness:.4f}")
                
                # Save fitness history to CSV
                with open("fitness_history_ANN.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Generation", "AverageFitness"])
                    for i, fitness in enumerate(self.fitness_history):
                        writer.writerow([i + 1, fitness])
                
                # Evolutionary algorithm
                parents = self.select_parents()
                new_population = parents.copy()
                
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
                
                self.population = new_population[:POPULATION_SIZE]
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted!")
            save = input("Do you want to save the current population? (y/n): ").lower()
            if save == 'y':
                filename = input("Enter filename to save (without extension): ") + ".npz"
                
                genomes = [ind['genome'] for ind in self.population]
                fitnesses = [ind['fitness'] for ind in self.population]
                
                np.savez(filename,
                         genomes=genomes,
                         fitnesses=fitnesses,
                         fitness_history=self.fitness_history)
                
                print(f"Population saved to {filename}")
                print(f"Fitness history saved to fitness_history_ANN.csv")
                
                # Display fitness graph data
                print("\nFitness history:")
                for gen, fitness in enumerate(self.fitness_history):
                    print(f"Generation {gen+1}: {fitness:.4f}")
            return

def main():
    controller = Evolution()
    controller.run()

if __name__ == "__main__":
    main()