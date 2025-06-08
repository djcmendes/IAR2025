#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#

import sys
import os
from datetime import datetime
import time
import gymnasium as gym
import numpy as np
import math
from torch import nn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO
from controller import Supervisor

MAX_SPEED = 9
NUM_OBS = 5
MAX_PS_VALUE = 4000.0    
MAX_GS_VALUE = 1000.0

#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps = 500, reward_config=None, random_obstacles=True):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', entry_point='openai_gym:OpenAIGymEnvironment', max_episode_steps=max_episode_steps)
        self.__timestep = int(self.getBasicTimeStep())
        self.max_episode_steps = max_episode_steps
        self.random_obstacles = random_obstacles

        self.reward_config = reward_config or {
            "penaliza_queda": True,
            "penaliza_proximidade": True,
            "recompensa_movimento": True,
            "recompensa_base": True
        }

        # Cada roda pode ter uma velocidade entre -1.0 e 1.0 (ajustável)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

        # 5 sensores frontais + 2 sensores de chão → 7 observações normalizadas
        self.observation_space = gym.spaces.Box(
            low=np.zeros(7), 
            high=np.ones(7), 
            dtype=np.float32
        )

        self.__n = 0
        self.state = None

        self.root = self.getRoot()
        self.robot_node = self.getFromDef("ROBOT")
        self.children_field = self.root.getField("children")

        # Get robot initial position
        self.init_translation_field = self.robot_node.getField("translation").getSFVec3f()
        self.last_position = self.init_translation_field

        # Setup sensors and motors
        self._setup_sensors()
        self._setup_motors()


    def _setup_sensors(self):
        """Initialize and enable all sensors"""
        self.ps = [self.getDevice(f'prox.horizontal.{i}') for i in range(5)]
        self.ground = [self.getDevice(f'prox.ground.{i}') for i in range(2)]
        for sensor in self.ps + self.ground:
            sensor.enable(self.__timestep)

    def _setup_motors(self):
        """Initialize and configure motors"""
        self.left_motor = self.getDevice('motor.left')
        self.right_motor = self.getDevice('motor.right')
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

    def _stop_motors(self):
        """Stop both motors"""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
    
    def _reset_pos(self):
        if self.robot_node is None:
            print("Erro: robô THYMIO não encontrado!")
        else:
            # Randomizar orientação (ângulo yaw)
            position = [0, 0, 1.01]     # centro do ambiente
            yaw = self.np_random.uniform(low=0, high=2*np.pi)
            rotation = [0, 0, 1, yaw]  # eixo z

            self.robot_node.getField("translation").setSFVec3f(position)
            self.robot_node.getField("rotation").setSFRotation(rotation)
            self.robot_node.resetPhysics()  # Reiniciar física do robô
    
    def _get_observation(self):
        """Get current sensor readings normalized to [0,1]"""
        readings = []
        
        # Front proximity sensors
        for sensor in self.ps:
            readings.append(sensor.getValue() / MAX_PS_VALUE)
            
        # Ground sensors
        for sensor in self.ground:
            readings.append(sensor.getValue() / MAX_GS_VALUE)
            
        return np.array(readings, dtype=np.float32)


    def _compute_reward(self, obs, action):
        reward = 0.0
        terminated = False
        info = {}

        current_position = self.robot_node.getField("translation").getSFVec3f()
        min_ground = min(obs[5:])  # Sensores de chão

        # ----- Queda (Z muito baixo ou buraco no chão) -----
        if self.reward_config.get("penaliza_queda", True):
            if min_ground < 0.6 or current_position[2] < (self.init_translation_field[2] - 0.1):
                reward -= 100
                terminated = True
                info['fall'] = True
                info['termination_reason'] = 'fall'
                return reward, terminated, info

        # ----- Distância percorrida (reforço positivo) -----
        dx = current_position[0] - self.last_position[0]
        dy = current_position[1] - self.last_position[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if self.reward_config.get("recompensa_movimento", True):
            reward += distance * 20  # mais forte que antes

        # Penalização por ficar parado muito tempo
        self.no_move_counter = getattr(self, 'no_move_counter', 0)
        if distance < 0.005:
            self.no_move_counter += 1
            if self.no_move_counter > 5:
                reward -= 5
        else:
            self.no_move_counter = 0

        # ----- Penalização progressiva por proximidade de obstáculos -----
        if self.reward_config.get("penaliza_proximidade", True):
            prox_penalty = sum([max(0, v - 0.5) for v in obs[:5]])  # penaliza a partir de 0.5
            reward -= prox_penalty * 10  # penalização proporcional

        # ----- Bónus por “sobrevivência” -----
        if self.reward_config.get("recompensa_base", True):
            reward += 0.5  # pequena recompensa constante por continuar sem falhar

        # ----- (Opcional) Exploração: novo espaço visitado -----
        # Podes implementar grid de células visitadas aqui para bónus adicional

        self.last_position = current_position
        return reward, terminated, info


    # def _compute_rewardv2(self, obs, action):
    #     reward = 0.0
    #     terminated = False
    #     info = {}

    #     current_position = self.robot_node.getField("translation").getSFVec3f()
    #     min_ground = min(obs[5:])  # Leitura dos sensores de chão

    #     # --- Queda (Z muito baixo ou leitura suspeita no sensor de chão) ---
    #     if self.reward_config.get("penaliza_queda", True):
    #         if min_ground < 0.7:
    #             reward -= 10

    #         if current_position[2] < (self.init_translation_field[2] - 0.1):
    #             terminated = True
    #             info['fall'] = True
    #             info['termination_reason'] = 'fall'
    #             return reward, terminated, info

    #     # --- Movimento: recompensa pela distância percorrida ---
    #     dx = current_position[0] - self.last_position[0]
    #     dy = current_position[1] - self.last_position[1]
    #     distance = math.sqrt(dx ** 2 + dy ** 2)

    #     if self.reward_config.get("recompensa_movimento", True):
    #         if distance < 0.005:
    #             self.no_move_counter = getattr(self, 'no_move_counter', 0) + 1
    #             if self.no_move_counter > 3:
    #                 reward -= 3
    #         else:
    #             self.no_move_counter = 0
    #             reward += distance * 15  # maior incentivo a explorar

    #     # --- Penalização proporcional à proximidade de obstáculos ---
    #     if self.reward_config.get("penaliza_proximidade", True):
    #         proximity_penalty = sum([max(0, v - 0.6) for v in obs[:5]])
    #         reward -= proximity_penalty * 5  # penaliza mais quando está muito próximo

    #     # --- Bónus extra se estiver a explorar (sem estar em risco) ---
    #     if distance > 0.01 and min_ground > 0.7:
    #         reward += 0.5

    #     # Guarda a nova posição
    #     self.last_position = current_position

    #     return reward, terminated, info


    # def _compute_rewardv1(self, obs, action):
        # reward = 0.0
        # terminated = False
        # info = {}

        # current_position = self.robot_node.getField("translation").getSFVec3f()
        # min_ground = min(obs[5:])  # Ground sensors

        # if self.reward_config.get("penaliza_queda", True) and min_ground < 0.7:
        #     reward -= 10

        # # Verificar queda real (Z abaixo do permitido)
        # if current_position[2] < (self.init_translation_field[2] - 0.1):
        #     terminated = True
        #     info['fall'] = True
        #     info['termination_reason'] = 'fall'
        #     return reward, terminated, info

        # dx = current_position[0] - self.last_position[0]
        # dy = current_position[1] - self.last_position[1]
        # distance = math.sqrt(dx ** 2 + dy ** 2)

        # if self.reward_config.get("recompensa_movimento", True):
        #     if distance < 0.01:
        #         reward -= 5
        #     else:
        #         reward += distance * 10

        # if self.reward_config.get("penaliza_proximidade", True):
        #     if max(obs[:5]) > 0.8:
        #         reward -= 1

        # if self.reward_config.get("recompensa_base", True):
        #     reward += 1

        # self.last_position = current_position
        # return reward, terminated, info

    def _randomize_obstacles(self):
        if not self.random_obstacles:
            return
        
        # Remove existing obstacles
        i = 0
        while i < self.children_field.getCount():
            node = self.children_field.getMFNode(i)
            node_def = node.getDef()
            if node_def and node_def.startswith("WHITE_BOX_"):
                self.children_field.removeMF(i)
            else:
                i += 1

        # Funções auxiliares
        def random_orientation():
            angle = self.np_random.uniform(0, 2 * np.pi)
            return (0, 0, 1, angle)  

        def random_position_on_H(exclusion_radius=0.2):
            while True:
                x = self.np_random.choice([
                    self.np_random.uniform(-0.3, -0.2),  # perna esquerda
                    self.np_random.uniform(0.2, 0.3),    # perna direita
                ])
                y = self.np_random.choice([
                    self.np_random.uniform(-0.5, -0.3),  # base
                    self.np_random.uniform(0.3, 0.5),    # topo
                ])
                if np.sqrt(x**2 + y**2) > exclusion_radius:
                    return x, y, 1.01

        # Criar novos obstáculos
        for i in range(NUM_OBS):
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
            self.children_field.importMFNodeFromString(-1, box_string)


    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.__timestep)

        # Reset counters and motors
        self.__n = 0
        self._stop_motors()
        self._reset_pos()
        self.last_position = self.robot_node.getField("translation").getSFVec3f()

        # you may need to iterate a few times to let physics stabilize
        for i in range(15):
            super().step(self.__timestep)

        # set the initial state vector to return
        init_state = self._get_observation()

        if self.random_obstacles:
            self._randomize_obstacles()

        return np.array(init_state).astype(np.float32), {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):
        self.__n += 1

        action = np.array(action, dtype=np.float32).flatten()
        left_speed = float(action[0] * MAX_SPEED)
        right_speed = float(action[1] * MAX_SPEED)
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        for _ in range(10):
            super().step(self.__timestep)

        self.state = self._get_observation()
        reward, terminated, info = self._compute_reward(self.state, action)

        truncated = self.__n >= self.max_episode_steps
        if truncated:
            reward += 200
            info['truncated'] = True

        return self.state.astype(np.float32), reward, terminated, truncated, info


def main():
    
    # Criar pasta única para este treino com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"./checkpoints/run_{timestamp}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create the environment to train / test the robot
    checkpoint = CheckpointCallback(save_freq=10000, save_path=checkpoint_path, name_prefix='ppo')
    env = make_vec_env(OpenAIGymEnvironment)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    env.reset()

    policy_kwargs = dict(
        activation_fn=nn.ReLU,  # ou nn.Tanh
        net_arch=[64, 32]       # ou outra arquitetura desejada
    )

    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cpu',
        learning_rate=3e-4, #default
        ent_coef = 0.01
    )

    print("Vai comecar o treino PPO")
    try:
        model.learn(total_timesteps=100000, callback=[checkpoint])
        model.save("ppo_thymio")
        env.save("ppo_thymio.vecnormalize")  # <- Adiciona esta linha
        print("Treino concluído e modelo guardado com sucesso.")

    except KeyboardInterrupt:
        print("Treino interrompido. A guardar modelo...")
        model.save("ppo_thymio_interrupt")


if __name__ == '__main__':
    main()
