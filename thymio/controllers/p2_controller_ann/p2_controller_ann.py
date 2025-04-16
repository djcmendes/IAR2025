import numpy as np
from controller import Supervisor

# Rede neuronal com 2 entradas, 1 camada escondida (4 neurónios), 2 saídas
class SimpleANN:
    def __init__(self, genome):
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 2

        # Extrair pesos e biases do genoma (tamanho total: 22)
        i = 0
        self.W1 = np.array(genome[i:i+8]).reshape((2, 4))
        i += 8
        self.b1 = np.array(genome[i:i+4])
        i += 4
        self.W2 = np.array(genome[i:i+8]).reshape((4, 2))
        i += 8
        self.b2 = np.array(genome[i:i+2])

    def forward(self, inputs):
        h = np.tanh(np.dot(inputs, self.W1) + self.b1)
        output = np.tanh(np.dot(h, self.W2) + self.b2)
        return output


class ANNController:
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep()*5)

        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        for s in self.ground_sensors:
            s.enable(self.timestep)

        self.EVALUATION_TIME = 300  # segundos
        self.collision = False

    def reset_position(self):
        self.robot_node.getField('rotation').setSFRotation([0, 0, 1, np.random.uniform(0, 2*np.pi)])
        self.robot_node.getField('translation').setSFVec3f([0, 0, 0])
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def normalize(self, value):
        return (value / 1023 - 0.6) / 0.2

    def runStep(self, ann):
        left_raw = self.ground_sensors[0].getValue()
        right_raw = self.ground_sensors[1].getValue()

        left_sensor = self.normalize(left_raw)
        right_sensor = self.normalize(right_raw)

        print(f"Raw: L={left_raw:.1f}, R={right_raw:.1f} | Norm: L={left_sensor:.2f}, R={right_sensor:.2f}")

        inputs = np.array([left_sensor, right_sensor])
        motor_speeds = ann.forward(inputs) * 9

        if np.all(inputs < 0.2):
            motor_speeds = [2.0, 2.0]

        self.left_motor.setVelocity(motor_speeds[0])
        self.right_motor.setVelocity(motor_speeds[1])
        self.supervisor.step(self.timestep)


    def run(self, genome):
        self.reset_position()
        ann = SimpleANN(genome)
        start_time = self.supervisor.getTime()

        while self.supervisor.getTime() - start_time < self.EVALUATION_TIME:
            self.runStep(ann)


# Exemplo de teste com genoma aleatório
def main():
    genome = np.random.uniform(-1, 1, 22)  # Tamanho do genoma: 22
    controller = ANNController()
    controller.run(genome)

if __name__ == "__main__":
    main()