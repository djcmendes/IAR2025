import numpy as np
from aiofiles.ospath import exists
from controller import Supervisor
import random
import csv

from tensorflow.python.keras.distribute.distributed_training_utils_v1 import set_weights

# Genoma inicial pré-definido (melhor comportamento conhecido)
default_genome = np.array([
    # W1 (2x4) - Pesos da camada de entrada para escondida
    1.0, -0.8, 0.5, -0.5, # Sensor esquerdo
    -0.8, 1.0, -0.5, 0.5, # Sensor direito
    0.5, -1.0, 0.5, -1.0, # Sensor 3 (prox. frontal esquerda)
    1.0, -0.5, 1.0, -0.5, # Sensor 4 (prox. frontal centro)
    -1.0, 0.5, -1.0, 0.5, # Sensor 5 (prox. frontal direita)

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


class AdvanceANN:
    """
    `Objetivo: Evoluir uma rede neuronal para seguimento de linha evitando obstáculos no meio da linha.
    o Inputs: 2 sensores de chão + 3 sensores de proximidade frontais (total: 5 entradas).`

    Rede neuronal com 5 entradas, 1 camada escondida (4 neurónios), 2 saídas
    """
    def __init__(self, genome):
        i = 0
        # sensores 2 chão + 3 proximidade sensores/inputs = 5, 4 neurónios = 20
        self.W1 = genome[i:i+20].reshape((5, 4)); i += 20
        self.b1 = genome[i:i+4]; i += 4
        self.W2 = genome[i:i+8].reshape((4, 2)); i += 8
        self.b2 = genome[i:i+2]

    def forward(self, inputs):
        h = np.tanh(np.dot(inputs, self.W1) + self.b1) # Activação da camada hidden layer 1
        output = np.tanh(np.dot(h, self.W2) + self.b2) # Saída final
        return output

class ANNController:
    def __init__(self, supervisor=None, evaluation_time=300):
        # Reuse Supervisor if fornecido, ou crie um
        self.supervisor = supervisor or Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field    = self.robot_node.getField("rotation")
        self.timestep = int(self.supervisor.getBasicTimeStep() * 5)

        self.set_weights()

        # Pesos iniciais para calculo do fitness
        # 1. seguir linha
        # 2. evitar obstaculos

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

        # 0 e 1 esquerda
        # 2 e 4 frente esquerda e direita
        # 3 centro
        # 5 e 6 direita
        self.prox_sensors = [
            self.supervisor.getDevice(f'prox.horizontal.{i}') for i in [0, 3, 6]
        ]
        for s in self.prox_sensors:
            s.enable(self.timestep)

        self.EVALUATION_TIME = evaluation_time
        self.time_in_line = 0

    def reset(self):
        self.rotation_field.setSFRotation([0,0,1, np.random.uniform(0,2*np.pi)])
        self.translation_field.setSFVec3f([0,0,0])
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.time_in_line = 0

        # Gerar obstáculos aleatórios — usa o código do Lab 1 aqui (a ser implementado por ti)
        self._randomize_obstacles()


    def set_weights(self, weights=None):
        if weights is None:
            weights = {}
        self.weights_follow          = weights.get("weights_follow", 0.5)
        self.weights_avoid_obstacles = weights.get("weights_avoid_obstacles", 0.4)
        self.weights_area            = weights.get("weights_area", 0.1)
        self.collision_penalty       = weights.get("collision_penalty", 0.0)
        self.off_line_penalty        = weights.get("off_line_penalty", 0.0)
        self.area_penalty            = weights.get("area_penalty", 0.0)

    def set_test_name(self, test):
        self.test = test

    def normalize_color(self, value):
        white, black = 1000.0, 500.0
        return max(0.0, min(1.0, (white - value)/(white - black)))

    def normalize_proximity(self, value):
        return max(0.0, min(1.0, value / 4096.0)) # 4096.0 maximo da reflexão detetada pelo sensor (é o valor mais proximo do obstaculo)

    def fitness(self, total_steps, time_without_collision, count_colisions, total_distance, max_possible_distance):
        score_color = (self.time_in_line / total_steps)
        score_proximity = (time_without_collision / total_steps)
        score_area = total_distance / max_possible_distance

        # garantir que fique entre 0 e 1
        score_color = min(1.0, max(0.0, score_color))
        score_proximity = min(1.0, max(0.0, score_proximity))
        score_area = min(1.0, max(0.0, score_area))

        collision_p = 0
        off_line_p = 0
        area_p = 0

        if score_color < 0.05:
            off_line_p = self.off_line_penalty
        if count_colisions > 10:
            collision_p= self.collision_penalty
        if score_area < 0.2:
            area_p = self.area_penalty


        fitness = (
            self.weights_follow * score_color +
            self.weights_avoid_obstacles * score_proximity +
            self.weights_area * score_area
        ) - (collision_p + off_line_p + area_p)

        fitness = max(fitness, 0)

        print("---------")
        print("fitness:", fitness)
        print("score_color:", score_color)
        print("score_proximity:", score_proximity)
        print("score_area:", score_area)
        print("collision_penalty:", collision_p)
        print("off_line_penalty:", off_line_p)
        print("area_penalty:", area_p)
        print("count_colisions:", count_colisions)

        """
        # função Loss
        loss = (
            self.weights_follow * (1 - score_color) +
            self.weights_avoid_obstacles * (1 - score_proximity) +
            collision_penalty + off_line_penalty
        )
        loss = max(0, min(loss, 1.0)) # clamp
        loss = loss * penalty_factor

        #loss = 0
        fitness = (
            self.weights_follow * score_color +
            self.weights_avoid_obstacles * score_proximity
        ) - (collision_penalty + off_line_penalty)

        # Ensure fitness is not negative
        fitness = max(fitness - loss, 0)

        print("---------")
        print("fitness:", fitness)
        print("fitness_color:", score_color)
        print("fitness_proximity:", score_proximity)
        print("collision_penalty:", collision_penalty)
        print("off_line_penalty:", off_line_penalty)
        print("count_colisions:", count_colisions)"""

        return fitness, score_color, score_proximity, score_area

    def run(self, genome):
        self.reset()
        ann = AdvanceANN(genome)
        start_time = self.supervisor.getTime()

        # medir a distancia percorrida
        prev_position = self.translation_field.getSFVec3f()
        total_distance = 0.0

        map_size = 3.0  # baseado no floor
        resolution = 0.05  # distancia por célula
        grid_size = int(map_size / resolution)
        visited_map = np.zeros((grid_size, grid_size), dtype=bool)

        time_without_collision = 0
        total_steps            = 0
        count_colisions        = 0
        max_speed              = 6
        while self.supervisor.getTime() - start_time < self.EVALUATION_TIME:
            s_ground = [self.normalize_color(sen.getValue()) for sen in self.ground_sensors]
            s_prox   = [self.normalize_proximity(s.getValue()) for s in self.prox_sensors]

            if s_ground[0] > 0.65 or s_ground[1] > 0.65:
                self.time_in_line += 1

            # Considera colisão se algum sensor de proximidade estiver muito ativo
            collision = any(p > 0.8 for p in s_prox)
            if not collision:
                time_without_collision += 1

            if any(p > 0.98 for p in s_prox):
                count_colisions += 1

            inputs = np.array(s_ground + s_prox)
            speeds = ann.forward(np.array(inputs)) * 6.0
            base = 3.0
            left  = max(0, min(max_speed, base + speeds[0]*2))
            right = max(0, min(max_speed, base + speeds[1]*2))
            self.left_motor.setVelocity(left)
            self.right_motor.setVelocity(right)
            self.supervisor.step(self.timestep)

            ## mediar distancia percorrida
            position = self.translation_field.getSFVec3f()
            x, _, z = position
            px, _, pz = prev_position

            # Calculate Euclidean distance in the XZ plane
            step_distance = ((x - px) ** 2 + (z - pz) ** 2) ** 0.5
            total_distance += self.exploration_bonus()

            prev_position = position # Update for next step

            total_steps += 1

        max_distance = round((max_speed * self.EVALUATION_TIME) / 0.1)
        fitness_score, score_color, score_proximity, score_area = self.fitness(total_steps, time_without_collision, count_colisions, total_distance, max_distance)

        return (fitness_score, score_color, score_proximity, score_area, count_colisions)

    def exploration_bonus(self):
        pos = self.robot_node.getField("translation").getSFVec3f()
        x, y = pos[0], pos[1]

        # Arredonda posição para "células de uma grelha virtual"
        grid_x = round(x / 0.1)
        grid_y = round(y / 0.1)

        key = (grid_x, grid_y)

        if not hasattr(self, "prev_visited_zones"):
            self.prev_visited_zones = set()

        if key not in self.prev_visited_zones:
            self.prev_visited_zones.clear() # remove all previous zones
            self.prev_visited_zones.add(key)
            return 1.0  # recompensa por nova zona
        else:
            return 0.0

    def _randomize_obstacles(self):
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
                # Gerar nas pernas do H (laterais) e nas travessas (topo/baixo)

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
        self.genome_len   = 34
        # Inicializa população com o genoma pré-definido em primeiro lugar
        self.population = [default_genome.copy()]
        for _ in range(self.pop_size - 1):
            self.population.append(
                np.random.uniform(-1, 1, self.genome_len)
            )
        self.fitnesses = np.zeros(self.pop_size)
        self.fitness_history = []

        print("....TEST ",self.controller.test)
        print("........Weights: ")
        print("...........weights_follow : ", self.controller.weights_follow)
        print("...........weights_avoid_obstacles : ", self.controller.weights_avoid_obstacles)
        print("...........weights_area : ", self.controller.weights_area)
        print("...........collision_penalty : ", self.controller.collision_penalty)
        print("...........off_line_penalty : ", self.controller.off_line_penalty)
        print("...........area_penalty : ", self.controller.area_penalty)


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
        # Cria um arquivo CSV para guardar a evolução do fitness (para analises)
        if self.controller.test:
            file_name = f"../../../data/fitness_history_ann_advanced_{self.controller.test}.csv"
        else:
            file_name = f"../../../data/fitness_history_ann_advanced.csv"

        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Individual", "Fitness", "Score_Color_Line", "Score_Proximity_Collisions", "Score_Distance_Area", "Collisions", "genome"])

            for gen in range(self.generations):
                print(f"=== Geração {gen+1}/{self.generations} ===")
                for i, genome in enumerate(self.population):
                    fit, color, proximity, area, collision  = self.controller.run(genome)

                    self.fitnesses[i] = fit
                    print(f"Ind {i+1}: fitness={fit:.3f}")

                    writer.writerow([gen + 1, i + 1, fit, color, proximity, area, collision, genome])  # guardar resultados no ficheiro
                    self.fitness_history.append((gen + 1, i + 1, fit, color, proximity, area, collision, genome))  #

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


    def run_genome(self, genome):
        # Cria um arquivo CSV para guardar a evolução do fitness (para analises)
        if self.controller.test:
            file_name = f"../../../data/fitness_history_ann_advanced_{self.controller.test}_genome.csv"
        else:
            file_name = f"../../../data/fitness_history_ann_advanced_genome.csv"

        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Individual", "Fitness", "Score_Color_Line", "Score_Proximity_Collisions", "Score_Distance_Area", "Collisions", "genome"])

            fit, color, proximity, area, collision  = self.controller.run(genome)

            writer.writerow([1, 1, fit, color, proximity, area, collision, genome])
            print(f"Fitness={fit:.3f}")
            print(f"Score black line={color:.3f}")
            print(f"Score time without collisions={proximity:.3f}")
            print(f"Score of How much distance roobot move={area:.3f}")
            print(f"Collision count={collision}")

if __name__ == "__main__":
    # Cria um único Supervisor e ANNController
    supervisor = Supervisor()

    weights = {
        "weights_follow": 0.5,
        "weights_avoid_obstacles": 0.4,
        "weights_area": 0.1,
        "collision_penalty": 0.0,
        "off_line_penalty": 0.0,
        "area_penalty": 0.0
    }

    controller = ANNController(
        supervisor=supervisor,
        #evaluation_time=5,
        evaluation_time=300,
    )

    controller.set_weights(weights)
    controller.set_test_name(1)

    # Evolução usando o mesmo controller
    evo = Evolution(controller, pop_size=10, generations=300, mutation_rate=0.2, mutation_scale=0.05)
    best_genome = evo.evolve()

    print("best_genome", best_genome)
    print("Run best")
    # Avaliação final sem recriar Supervisor
    evo.run_genome(best_genome)

    print("--------------1 test done--------------------")

    weights = {
        "weights_follow": 0.6,
        "weights_avoid_obstacles": 0.3,
        "weights_area": 0.1,
        "collision_penalty": 0.2,
        "off_line_penalty": 0.2,
        "area_penalty": 0.2
    }

    controller.set_weights(weights)
    controller.set_test_name(2)

    # Evolução usando o mesmo controller
    evo = Evolution(controller, pop_size=10, generations=300, mutation_rate=0.2, mutation_scale=0.05)
    best_genome = evo.evolve()

    # Avaliação final sem recriar Supervisor
    #evo.controller.run(best_genome)
    evo.run_genome(best_genome)

    print("--------------2 test done--------------------")

    supervisor = Supervisor()

    weights = {
        "weights_follow": 0.6,
        "weights_avoid_obstacles": 0.3,
        "weights_area": 0.1,
        "collision_penalty": 0.4,
        "off_line_penalty": 0.4,
        "area_penalty": 0.4
    }

    controller.set_weights(weights)
    controller.set_test_name(3)

    # Evolução usando o mesmo controller
    evo = Evolution(controller, pop_size=10, generations=300, mutation_rate=0.2, mutation_scale=0.05)
    best_genome = evo.evolve()

    # Avaliação final sem recriar Supervisor
    #controller.run(best_genome)
    evo.run_genome(best_genome)

    print("--------------3 test done--------------------")