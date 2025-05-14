#
# ISCTE-IUL, IAR, 2024/2025.
#
# PPO Training for Thymio Robot in Webots.
#

import tensorflow
print("TensorFlow imported successfully")
try:
    import tensorboard
    print("TensorBoard imported successfully")
except ImportError as e:
    print(f"Error importing TensorBoard: {e}")
except AttributeError as e:
    print(f"AttributeError related to TensorBoard/Protobuf: {e}")

import sys

try:
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
        actual_left_vel = float(np.clip(left_speed, -1, 1)) * MAX_SPEED
        actual_right_vel = float(np.clip(right_speed, -1, 1)) * MAX_SPEED
        
        # print(f"Step {self.__n}: Setting motor vels = [{actual_left_vel:.4f}, {actual_right_vel:.4f}]") # DEBUG
        
        self.left_motor.setVelocity(actual_left_vel)
        self.right_motor.setVelocity(actual_right_vel)

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

def test():
    print("Starting test() for basic robot movement...")
    
    # 1. Create an instance of your environment.
    # The max_episode_steps doesn't really matter for this direct control test.
    robot_controller = OpenAIGymEnvironment(max_episode_steps=100)

    # 2. Ensure motors are available (they should be initialized in __init__)
    if robot_controller.left_motor is None or robot_controller.right_motor is None:
        print("CRITICAL in test(): Motors not found. Aborting test.")
        return
    print("Motors appear to be initialized in test().")

    # 3. Define a speed for the motors (e.g., rad/s)
    # MAX_SPEED_WEBOTS in your class is 6.28. Let's try a moderate speed.
    test_forward_speed = 3.0  # rad/s
    print(f"Setting motor velocities to: {test_forward_speed} rad/s")
    robot_controller.left_motor.setVelocity(test_forward_speed)
    robot_controller.right_motor.setVelocity(test_forward_speed)

    # 4. Let the simulation run for a certain number of Webots steps
    # Access the private __timestep variable via its mangled name for this test
    # This is generally not good practice but acceptable for a direct test script.
    webots_timestep_duration = robot_controller._OpenAIGymEnvironment__timestep
    if webots_timestep_duration <= 0:
        print(f"CRITICAL in test(): Invalid Webots timestep: {webots_timestep_duration}. Aborting.")
        return
        
    simulation_run_steps = 200  # Run for 200 Webots basic timesteps
    print(f"Running Webots simulation for {simulation_run_steps} steps of {webots_timestep_duration}ms each...")

    for i in range(simulation_run_steps):
        # Call the step method of the SUPERVISOR class directly.
        # This advances the Webots simulation by one basic timestep.
        # `robot_controller` is an instance of OpenAIGymEnvironment, which is a Supervisor.
        # `Supervisor.step(instance_of_supervisor, duration_of_step)`
        if Supervisor.step(robot_controller, webots_timestep_duration) == -1:
            print(f"Webots simulation terminated by an external event at step {i+1}.")
            break
        
        # Optional: Print a message periodically
        if (i + 1) % 50 == 0:
            print(f"Webots simulation step {i + 1} / {simulation_run_steps} completed.")
            # You could also get sensor readings here if needed for debugging:
            # current_sensors = robot_controller._get_sensor_state()
            # print(f"Current sensors: {current_sensors}")

    print("Test movement duration finished.")

    # 5. Stop the motors
    print("Stopping motors...")
    robot_controller.left_motor.setVelocity(0.0)
    robot_controller.right_motor.setVelocity(0.0)
    # Let the stop command process for a few steps
    for _ in range(50):
        if Supervisor.step(robot_controller, webots_timestep_duration) == -1:
            break
            
    print("Robot stop command sent. test() finished.")
    print("Please check the Webots window to see if the robot moved.")

if __name__ == '__main__':
    main()
    #test()

