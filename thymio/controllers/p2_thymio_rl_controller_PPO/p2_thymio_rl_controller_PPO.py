#
# ISCTE-IUL, IAR, 2024/2025.
#
# PPO Training for Thymio Robot in Webots.
#

try:
    import sys
    import time
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from controller import Supervisor

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


#
# Structure of a class to create an OpenAI Gym in Webots.
#

class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps = ...):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(
            id='WebotsEnv-v0', 
            entry_point='openai_gym:OpenAIGymEnvironment', 
            max_episode_steps=max_episode_steps
        )
        self.__timestep = int(self.getBasicTimeStep())
        self.__n = 0

        # Espaço de ações: velocidades [-1.0, 1.0] para cada roda
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Espaço de observações: 5 sensores proximidade + 2 sensores de chão
        self.observation_space = gym.spaces.Box(
            low=np.zeros(7),
            high=np.ones(7),
            dtype=np.float32
        )

        # Sensores
        self.prox_sensors = [self.getDevice(f"prox.horizontal.{i}") for i in range(5)]
        self.ground_sensors = [self.getDevice(f"prox.ground.{i}") for i in range(2)]
        for sensor in self.prox_sensors + self.ground_sensors:
            sensor.enable(self.__timestep)

        # Motores
        self.left_motor = self.getDevice("motor.left")
        self.right_motor = self.getDevice("motor.right")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.boxes = [self.getFromDef(f"OBS_{i}") for i in range(3)]

        # Histórico de posições visitadas
        self.visited = set()


    def _get_sensor_state(self):
        prox_readings = [sensor.getValue() for sensor in self.prox_sensors]
        ground_readings = [sensor.getValue() for sensor in self.ground_sensors]

        # Normalização simples
        norm_prox = [min(p / 4000.0, 1.0) for p in prox_readings]
        norm_ground = [min(g / 4000.0, 1.0) for g in ground_readings]

        return np.array(norm_prox + norm_ground)

    def _randomize_obstacles(self):
        for box in self.boxes:
            if box is None:
                continue
            trans = box.getField("translation")
            rot = box.getField("rotation")

            # evitar zona central onde nasce o robô
            x = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(-0.4, 0.4)
            while abs(x) < 0.1 and abs(z) < 0.1:
                x = np.random.uniform(-0.4, 0.4)
                z = np.random.uniform(-0.4, 0.4)

            trans.setSFVec3f([x, 0.02, z])
            rot.setSFRotation([0, 1, 0, np.random.uniform(0, 2*np.pi)])
    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)
        self.__n = 0

        # Motores parados
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Obstáculos aleatórios
        self._randomize_obstacles()
        self.visited = set()

        for i in range(15):  # estabilizar física
            super().step(self.__timestep)

        self.state = self._get_sensor_state()
        return self.state.astype(np.float32), {}

    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):
        self.__n += 1
        left_speed, right_speed = action

        MAX_SPEED = 6.28
        self.left_motor.setVelocity(float(np.clip(left_speed, -1, 1)) * MAX_SPEED)
        self.right_motor.setVelocity(float(np.clip(right_speed, -1, 1)) * MAX_SPEED)

        for _ in range(10):
            super().step(self.__timestep)

        self.state = self._get_sensor_state()

        # Atualiza posição do robô
        robot_node = self.getSelf()
        self.robot_position = robot_node.getField("translation").getSFVec3f()

        # (1) Evitar precipícios
        ground_safe = 1.0 - np.mean(self.state[5:])

        # (2) Evitar obstáculos
        obstacle_penalty = np.mean(self.state[:5])

        # (3) Explorar espaço (grid virtual)
        x, z = self.robot_position[0], self.robot_position[2]
        cell = (round(x, 1), round(z, 1))
        new_area_reward = 1.0 if cell not in self.visited else 0.0
        self.visited.add(cell)

        # (4) Preferir movimento para a frente
        linear_velocity = (left_speed + right_speed) / 2
        velocity_reward = max(0.0, linear_velocity)

        # (5) Recompensa total ponderada
        reward = (
            + 2.0 * ground_safe
            - 1.0 * obstacle_penalty
            + 1.5 * new_area_reward
            + 1.0 * velocity_reward
        )

        terminated = bool(np.any(self.state[5:] < 0.1))  # chão inseguro
        truncated = bool(self.__n >= 200)

        return self.state.astype(np.float32), reward, terminated, truncated, {}


def main():
    env = OpenAIGymEnvironment(max_episode_steps=200)
    check_env(env)  # opcional: verifica erros comuns

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)  # ajustar conforme testes
    model.save("ppo_thymio")

if __name__ == '__main__':
    main()

