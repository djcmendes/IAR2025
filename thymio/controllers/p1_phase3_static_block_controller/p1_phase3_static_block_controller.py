
from controller import Supervisor, Robot
import numpy as np
import random

class StaticBlockController:
    reset_flag = False

    def __init__(self, supervisor=None):
        self.supervisor = supervisor or Supervisor()
        self.robot = Robot()
        self.reset()

    def reset(self):
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.translation_field = self.supervisor.getSelf().getField("translation")

        # baseado no floor
        size = 1.5
        ignore_lines_prob = 0.2
        if random.random() < ignore_lines_prob:
            start_pos = [np.random.uniform(-size, size), 0.3,  np.random.uniform(-size, size)]
        else:
            # baseado no .wbt
            black_lines = [
                [1.0, 0.01, 1.0],
                [0.99, 0.01, 0.03],
                [-1.06, 0.01, -0.03],
                [0.05, 0.01, 0.99],
                [-0.95, 0.01, 0.88],
                [0.86, 0.01, -0.97],
                [-0.94, 0.01, -0.94]
            ]

            # Escolhe uma linha aleatória e aplica pequeno desvio
            base_pos = random.choice(black_lines)
            offset = np.random.uniform(-1.4, 1.4, size=2)

            start_pos = [base_pos[0] + offset[0], 0.02, base_pos[2] + offset[1]]

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

if __name__ == "__main__":
    # Cria um único Supervisor e ANNController
    supervisor = Supervisor()
    controller = StaticBlockController(supervisor)

    controller.run()
