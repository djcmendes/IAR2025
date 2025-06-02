
from controller import Supervisor, Robot
import numpy as np
import random

class MovingBlockController:
    reset_flag = False

    def __init__(self, supervisor=None):
        self.max_velocity = None
        self.velocity = None
        self.pause_counter = None
        self.supervisor = supervisor or Supervisor()
        self.robot = Robot()
        self.reset()

    def reset(self):
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.translation_field = self.supervisor.getSelf().getField("translation")

        self.max_velocity = 0.0
        if random.random() < 0.5:
            self.max_velocity = 0.001

        self.velocity = np.random.uniform(0, self.max_velocity, size=2)  # velocidade inicial aleatória

        self.pause_counter = 0

        # baseado no floor
        size = 1.5

        ignore_lines_prob = 0.4

        if random.random() < ignore_lines_prob:
            start_pos = [np.random.uniform(-size, size), 0.012, np.random.uniform(-size, size)]
        else:
            # baseado no .wbt
            black_lines = [
                [1.0, 0.012, 1.0],
                [0.99, 0.012, 0.03],
                [-1.06, 0.012, -0.03],
                [0.05, 0.012, 0.99],
                [-0.95, 0.012, 0.88],
                [0.86, 0.012, -0.97],
                [-0.94, 0.012, -0.94]
            ]

            # Escolhe uma linha aleatória e aplica pequeno desvio
            base_pos = random.choice(black_lines)
            offset = np.random.uniform(-1.4, 1.4, size=2)

            x = max(-1.4, min(1.4, base_pos[0] + offset[0]))
            z = max(-1.4, min(1.4, base_pos[2] + offset[1]))

            if x < 0.3 and x > -0.3:
                x = 0.35 if x >= 0 else -0.35
            if z < 0.3 and z > -0.3:
                z = 0.35 if z >= 0 else -0.35

            start_pos = [x, base_pos[1], z]

        x = max(-1.4, min(1.4, start_pos[0]))
        y = max(-1.4, min(1.4, start_pos[2]))
        if x < 0.3 and x > -0.3:
            x = 0.35 if x >= 0 else -0.35
        if y < 0.3 and y > -0.3:
            y = 0.35 if y >= 0 else -0.35

        start_pos = [x, start_pos[1], y]

        self.translation_field.setSFVec3f(start_pos)

    def run(self):
        while self.supervisor.step(self.timestep) != -1:
            custom_data = self.robot.getCustomData()
            if custom_data == "reset":
                # Executar a função reset
                #print("Resetting block...")
                self.robot.setCustomData("") # Limpa o sinal após o reset
                self.reset()

            pos = self.translation_field.getSFVec3f()

            # Se estiver em pausa, conta até sair
            if self.pause_counter > 0:
                self.pause_counter -= 1
                continue

            # Movimento
            new_pos = [pos[0] + self.velocity[0], pos[1], pos[2] + self.velocity[1]]

            # Colisão com limites do cenário (ajusta conforme o teu mundo)
            if new_pos[0] < -0.5 or new_pos[0] > 0.5:
                self.velocity[0] = np.random.uniform(-self.max_velocity, self.max_velocity)
            if new_pos[2] < -0.5 or new_pos[2] > 0.5:
                self.velocity[1] = np.random.uniform(-self.max_velocity, self.max_velocity)

            # Atualiza posição
            self.translation_field.setSFVec3f(new_pos)

            # Chance de parar aleatoriamente (simula paragem sobre a linha)
            if random.random() < 0.01: # 1% de chance por passo
                self.pause_counter = random.randint(10, 50) # pausa entre 10 e 50 passos

if __name__ == "__main__":
    # Cria um único Supervisor e ANNController
    controller = MovingBlockController(Supervisor())

    controller.run()
