import numpy as np
from controller import Supervisor
import random
import math
import csv

# Simulation parameters
TIME_STEP = 64
POPULATION_SIZE = 10
PARENTS_KEEP = 3
INPUT = 2
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (INPUT * HIDDEN) + HIDDEN + (HIDDEN * OUTPUT) + OUTPUT
GENERATIONS = 3000
MUTATION_RATE = 0.1
MUTATION_SIZE = 0.4
EVALUATION_TIME = 200  # seconds
MAX_SPEED = 6.28
BASE_SPEED = 3.0  # always forward

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

        # Start with a new population
        self.population = [{'genome': np.random.uniform(-1, 1, GENOME_SIZE), 'fitness': 0}
                           for _ in range(POPULATION_SIZE)]
        
    def reset(self):
        self.robot_node.resetPhysics()
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.time_in_line = 0
        self.step_count = 0
        self.collision = False

        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.rotation_field.setSFRotation(random_rotation)
        pos = random_position(-1, 1, 0.05)
        self.translation_field.setSFVec3f(pos)

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

    def mutate(self, genome):
        for i in range(len(genome)):
            if random.random() < MUTATION_RATE:
                genome[i] += np.random.normal(0, MUTATION_SIZE)
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

        # Sensor collision detection disabled (not used here)
        # Collision logic could be added if needed

        sensor_values = [((s.getValue() / 1023.0) - 0.5) * 2 for s in self.ground_sensors]

        ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - 0.6)/0.2 > 0.3
        ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - 0.6)/0.2 > 0.3
        
        # Corrected condition order
        if not ground_sensor_left and not ground_sensor_right:
            self.time_in_line += 10  # Higher reward for centered
        elif not ground_sensor_left or not ground_sensor_right:
            self.time_in_line += 5   # Lower reward for partial contact

        W1, b1, W2, b2 = self.decode_genome(genome)
        hidden = np.tanh(np.dot(sensor_values, W1) + b1)
        output = np.tanh(np.dot(hidden, W2) + b2)

        left_speed = BASE_SPEED + abs(output[0]) * (MAX_SPEED - BASE_SPEED)
        right_speed = BASE_SPEED + abs(output[1]) * (MAX_SPEED - BASE_SPEED)

        

        #if self.step_count % 100 == 0:
        #    print (f" leftspeed: {left_speed}, right_speed: {right_speed}")

        self.left_motor.setVelocity(max(min(left_speed, MAX_SPEED), 0))
        self.right_motor.setVelocity(max(min(right_speed, MAX_SPEED), 0))

        self.supervisor.step(self.timestep)

    def run(self):
        start_gen = len(self.fitness_history)

        try:
            for gen in range(start_gen, GENERATIONS):
                print(f"\n=== Generation {gen + 1}/{GENERATIONS} ===")
                
                for idx, individual in enumerate(self.population):
                    fitness_runs = []  # Store fitness for each run
                    
                    for run_num in range(3):  # Perform 3 runs per individual
                        self.reset()
                        self.evaluation_start_time = self.supervisor.getTime()
                        
                        print(f"\r  Individual {idx+1}/{POPULATION_SIZE} - Run {run_num+1}/3", 
                              end="", flush=True)
                        
                        # Reset run-specific metrics
                        self.time_in_line = 0
                        
                        # Run simulation for EVALUATION_TIME
                        while (self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME):
                            self.run_step(individual['genome'])
                            
                        # Calculate fitness for this run
                        fitness_run = self.time_in_line / EVALUATION_TIME
                        old_avg_fitness = sum(ind['fitness'] for ind in self.population) / POPULATION_SIZE
                        # print(f" old avg: {old_avg_fitness} ", end ="")
                        fitness_run = self.time_in_line / EVALUATION_TIME
                        if fitness_run > old_avg_fitness + 9:
                            print(f" - Invalid fitness_run {fitness_run:.4f} > {old_avg_fitness:.4f} + 9: setting to 0 ", end = "")
                            fitness_run = 0.0
                        fitness_runs.append(fitness_run)
                    
                    # Calculate average fitness across 3 runs
                    avg_fitness = sum(fitness_runs) / len(fitness_runs)
                    individual['fitness'] = avg_fitness
                    print(f" - Avg Fitness: {avg_fitness:.4f} (Runs: {fitness_runs})")

                avg_fitness = sum(ind['fitness'] for ind in self.population) / POPULATION_SIZE
                self.fitness_history.append(avg_fitness)
                print(f"Average fitness: {avg_fitness:.4f}")

                with open("fitness_history_ANN.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Generation", "AverageFitness"])
                    for i, fitness in enumerate(self.fitness_history):
                        writer.writerow([i + 1, fitness])

                parents = self.select_parents()
                new_population = parents.copy()

                while len(new_population) < POPULATION_SIZE:
                    parent1, parent2 = random.sample(parents, 2)
                    child1_genome, child2_genome = self.crossover(parent1['genome'], parent2['genome'])
                    child1 = {'genome': self.mutate(child1_genome), 'fitness': 0}
                    child2 = {'genome': self.mutate(child2_genome), 'fitness': 0}
                    new_population.extend([child1, child2])

                best_individual = max(self.population, key=lambda ind: ind['fitness']) # elitism
                self.population = [best_individual] + new_population[:POPULATION_SIZE-1]

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

def main():
    controller = Evolution()
    controller.run()

if __name__ == "__main__":
    main()