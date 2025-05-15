#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#

import sys

import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from controller import Supervisor

MAX_SPEED = 6.28 
NUM_OBSTACLES = 5

#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=500):
        super().__init__()

        self.spec = gym.envs.registration.EnvSpec(
            id='WebotsEnv-v0',
            entry_point='openai_gym:OpenAIGymEnvironment',
            max_episode_steps=max_episode_steps
        )

        self.__timestep = int(self.getBasicTimeStep())

        # Espaço de ações: velocidades angulares das duas rodas [-1.0, 1.0]
        # Estes valores serão escalados dentro do controlador
        self.action_space = gym.spaces.Box(
            low = -1,  
            high = 1,
            shape=(2,),
            dtype=np.float32
        )

    
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(7,),
            dtype=np.float32
        )

        # Inicializações internas
        self.state = None
        self.__n = 0

        # Aceder aos sensores
        self.ps = [self.getDevice(f'prox.horizontal.{i}') for i in range(5)]
        self.ground = [self.getDevice(f'prox.ground.{i}') for i in range(2)]
        for sensor in self.ps + self.ground:
            sensor.enable(self.__timestep)

        # Aceder aos motores
        self.left_motor = self.getDevice('motor.left')
        self.right_motor = self.getDevice('motor.right')
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

    def _get_observation(self):
        readings = []

        # Sensores frontais
        for sensor in self.ps:
            raw = sensor.getValue()
            norm = raw / 4096.0  # ajusta se necessário
            readings.append(np.clip(norm, 0.0, 1.0))

        # Sensores de chão
        for sensor in self.ground:
            raw = sensor.getValue()
            norm = raw / 1000.0  # ajusta se necessário
            readings.append(np.clip(norm, 0.0, 1.0))

        return np.array(readings, dtype=np.float32)

    def _compute_reward(self, obs, action):
        front_proximity = np.mean(obs[:5])  # sensores frontais
        ground_reading = np.mean(obs[5:])   # sensores de chão
        linear_velocity = (action[0] + action[1]) / 2.0

        reward = 0.0
        reward -= front_proximity * 2.0      # penaliza proximidade
        reward -= (1 - ground_reading) * 5.0 # penaliza risco de queda
        reward += linear_velocity            # prefere movimento para a frente

        return reward

    def _check_termination(self, obs):
        ground_reading = np.mean(obs[5:])
        return ground_reading < 0.1  # ou ajusta conforme o sensor


    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)

        # Reiniciar contagem de passos
        self.__n = 0

        # Motores parados
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Reposicionar o robô no centro com orientação aleatória
        thymio_node = self.getFromDef("ROBOT")  # o robô deve ter DEF nomeado "THYMIO" no .wbt
        if thymio_node is None:
            print("Erro: robô THYMIO não encontrado!")
        else:
            # Randomizar orientação (ângulo yaw)
            position = [0, 0, 1.01]     # centro do ambiente
            yaw = self.np_random.uniform(low=0, high=np.pi)
            rotation = [0, 0, 1, yaw]  # eixo Y

            thymio_node.getField("translation").setSFVec3f(position)
            thymio_node.getField("rotation").setSFRotation(rotation)
        # thymio_node.resetPhysics()  # Reiniciar física do robô

        # Gerar obstáculos aleatórios — usa o código do Lab 1 aqui (a ser implementado por ti)
        # self._randomize_obstacles()

        # Deixar a física estabilizar
        for _ in range(15):
            super().step(self.__timestep)

        # Ler sensores para estado inicial
        init_state = self._get_observation()

        return np.array(init_state).astype(np.float32), {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):
        self.__n += 1

        # 1. Aplicar ação nos motores (velocidades angulares)
        self.left_motor.setVelocity(action[0] * MAX_SPEED)
        self.right_motor.setVelocity(action[1] * MAX_SPEED)

        # 2. Deixar o efeito da ação decorrer por alguns passos
        for _ in range(10):
            super().step(self.__timestep)

        # 3. Observar o novo estado
        self.state = self._get_observation()

        # 4. Calcular recompensa
        reward = self._compute_reward(self.state, action)

        # 5. Verificar condições de paragem
        terminated = self._check_termination(self.state)
        truncated = self.__n >= self.spec.max_episode_steps

        print(f"Step {self.__n}: Action = {action}, Reward = {reward:.4f}, Terminated = {terminated}, Truncated = {truncated}")

        return self.state.astype(np.float32), reward, terminated, truncated, {}

def main():

    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()
    
    # check_env(env)  # opcional: verifica erros comuns

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)  # ajustar conforme testes
    model.save("ppo_thymio")
    


if __name__ == '__main__':
    main()
