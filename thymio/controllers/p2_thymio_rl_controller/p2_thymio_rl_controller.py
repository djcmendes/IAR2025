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
from sb3_contrib import RecurrentPPO
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

    # def _compute_reward(self, obs, action):
    #     front_proximity = np.mean(obs[:5])  # sensores frontais
    #     ground_reading = np.mean(obs[5:])   # sensores de chão
    #     linear_velocity = (action[0] + action[1]) / 2.0

    #     reward = 0.0
    #     reward -= front_proximity * 2.0      # penaliza proximidade
    #     reward -= (1 - ground_reading) * 5.0 # penaliza risco de queda
    #     reward += linear_velocity            # prefere movimento para a frente

    #     return reward^

    def _exploration_bonus(self):
        pos = self.getFromDef("ROBOT").getField("translation").getSFVec3f()
        x, y = pos[0], pos[1]

        # Arredonda posição para "células de uma grelha virtual"
        grid_x = round(x / 0.1)
        grid_y = round(y / 0.1)

        if not hasattr(self, "visited_zones"):
            self.visited_zones = set()

        key = (grid_x, grid_y)
        if key not in self.visited_zones:
            self.visited_zones.add(key)
            return 1.0  # recompensa por nova zona
        else:
            return 0.0


    def _compute_reward(self, obs, action):
        # 1. Penalizar proximidade com obstáculos
        front_proximity = np.mean(obs[:5])
        reward = -2.0 * front_proximity

        # 2. Penalizar risco de queda (valores baixos nos sensores de chão)
        ground_reading = np.mean(obs[5:])
        reward -= 5.0 * (1 - ground_reading)

        # 3. Recompensar velocidades positivas (movimento em frente)
        linear_velocity = (action[0] + action[1]) / 2.0
        reward += linear_velocity  # podes ponderar com 1.5, 2.0...

        # 4. Penalizar estar parado (evita que fique a "pensar")
        if abs(action[0]) < 0.1 and abs(action[1]) < 0.1:
            reward -= 0.5

        # 5. Penalizar rodar sobre si próprio sem avançar
        if np.sign(action[0]) != np.sign(action[1]):
            reward -= 0.2

        # 6. (Opcional) Recompensar exploração de novas zonas
        reward += self._exploration_bonus()

        return reward


    def _check_termination(self, obs):
        ground_reading = np.mean(obs[5:])
        return ground_reading < 0.1  # ou ajusta conforme o sensor

    def _randomize_obstacles(self):
    # Aceder ao campo children da raiz da cena
        root = self.getRoot()
        children_field = root.getField("children")

        # Eliminar caixas anteriores (limpeza opcional)
        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)
            if node.getDef() and node.getDef().startswith("WHITE_BOX_"):
                children_field.removeMF(i)
                break  # necessário reiniciar contagem se remove (ver abaixo)

        # Gerar novos obstáculos
        def random_orientation():
            angle = self.np_random.uniform(0, 2 * np.pi)
            return (0, 1, 0, angle)  # rotação no eixo Y

        def random_position_on_H(exclusion_radius=0.2):
            while True:
                # Gerar nas pernas do H (laterais) e nas travessas (topo/baixo)
                x = self.np_random.choice([
                    self.np_random.uniform(-0.3, -0.2),  # perna esquerda
                    self.np_random.uniform(0.2, 0.3),    # perna direita
                ])
                y = self.np_random.choice([
                    self.np_random.uniform(-0.5, -0.3),  # parte inferior do H
                    self.np_random.uniform(0.3, 0.5),    # parte superior do H
                ])
                # Evita gerar no centro
                if np.sqrt(x**2 + y**2) > exclusion_radius:
                    return x, y, 1.01  # z fixo (altura do obstáculo)



        for i in range(5):
            position = random_position_on_H()
            orientation = random_orientation()
            length = self.np_random.uniform(0.05, 0.2)
            width = self.np_random.uniform(0.05, 0.2)

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
        self.visited_zones = set()
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
        thymio_node.resetPhysics()  # Reiniciar física do robô

        # Gerar obstáculos aleatórios — usa o código do Lab 1 aqui (a ser implementado por ti)
        self._randomize_obstacles()

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

        # print(f"Step {self.__n}: Action = {action}, Reward = {reward:.4f}, Terminated = {terminated}, Truncated = {truncated}")

        return self.state.astype(np.float32), reward, terminated, truncated, {}

def main():

    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()
    
    # check_env(env)  # opcional: verifica erros comuns

    model = PPO(
        "MlpPolicy", #"MlpPolicy", 
        env, 
        ent_coef=0.1,
        gamma=0.99,
        verbose=1)
    model.learn(total_timesteps=50000)  # ajustar conforme testes
    model.save("ppo_thymio")
    


if __name__ == '__main__':
    main()
