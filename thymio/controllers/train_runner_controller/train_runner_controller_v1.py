import os
import sys
import logging
from datetime import datetime
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# === CONFIGURAÇÕES DO TREINO ===
USE_LSTM = False                  # True = RecurrentPPO, False = PPO
ACTIVATION = nn.ReLU            # nn.ReLU ou nn.Tanh
RANDOM_OBSTACLES = True          # True = obstáculos aleatórios
USE_PENALTIES = True             # True = aplicar penalizações
TOTAL_TIMESTEPS = 300000         # Timesteps de treino

# === IMPORTAÇÃO DO AMBIENTE ===
if USE_LSTM:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'p2_thymio_rl_controller_RecurrentPPO'))
    from p2_thymio_rl_controller_RecurrentPPO import OpenAIGymEnvironment
    model_class = RecurrentPPO
    policy = 'MlpLstmPolicy'
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'p2_thymio_rl_controller_PPO'))
    from p2_thymio_rl_controller_PPO import OpenAIGymEnvironment
    model_class = PPO
    policy = 'MlpPolicy'

# === NOMES E PASTAS ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f"{'R' if USE_LSTM else ''}PPO_{'ReLU' if ACTIVATION==nn.ReLU else 'Tanh'}_" \
         f"{'randObs' if RANDOM_OBSTACLES else 'fixedObs'}_" \
         f"{'penalty' if USE_PENALTIES else 'noPenalty'}"

checkpoint_path = f"./checkpoints/{suffix}_{timestamp}"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/{suffix}_{timestamp}.log"

# === LOGGING CONFIGURATION ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

file_handler = logging.FileHandler(log_filename, encoding='cp1252')
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.__stdout__)
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# === REDIRECIONAR stdout/stderr para logger ===
class StreamToLogger:
    def __init__(self, logger_func):
        self.logger_func = logger_func
    def write(self, message):
        message = message.strip()
        if message:
            self.logger_func(message)
    def flush(self): pass

sys.stdout = StreamToLogger(logger.info)
sys.stderr = StreamToLogger(logger.error)

# === CONFIGURAÇÃO DA FUNÇÃO DE RECOMPENSA ===
reward_config = {
    "penaliza_queda": USE_PENALTIES,
    "penaliza_proximidade": USE_PENALTIES,
    "recompensa_movimento": True,
    "recompensa_base": True
}

# === CRIAR AMBIENTE ===
def env_fn():
    return OpenAIGymEnvironment(
        reward_config=reward_config,
        random_obstacles=RANDOM_OBSTACLES
    )

env = make_vec_env(env_fn)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env.reset()

# === DEFINIÇÃO DO MODELO ===
policy_kwargs = dict(
    activation_fn=ACTIVATION,
    net_arch=[64, 32]
)

model = model_class(
    policy,
    env,
    verbose=1,
    device='cpu',
    learning_rate=3e-4,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs
)

checkpoint = CheckpointCallback(save_freq=10000, save_path=checkpoint_path, name_prefix='thymio')

print(f"Iniciando treino do modelo: {suffix}")
print(f"Algoritmo: {'RecurrentPPO' if USE_LSTM else 'PPO'} | Ativação: {ACTIVATION.__name__} | "
      f"Obstáculos: {'Aleatórios' if RANDOM_OBSTACLES else 'Fixos'} | Penalizações: {USE_PENALTIES}")

try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint])
    model.save(f"{suffix}_{timestamp}_model")
    env.save(f"{suffix}_{timestamp}_vecnormalize")
    print("Modelo treinado e guardado com sucesso.")

except KeyboardInterrupt:
    print("Treino interrompido manualmente. A guardar modelo...")
    model.save(f"{suffix}_{timestamp}_interrupt")

