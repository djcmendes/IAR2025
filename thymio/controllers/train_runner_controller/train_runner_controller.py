import os
import sys
import logging
from controller import Supervisor
from datetime import datetime
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback


# Adiciona os paths relativos aos controladores
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '..', 'p2_thymio_rl_controller_PPO'))
sys.path.append(os.path.join(base_dir, '..', 'p2_thymio_rl_controller_RecurrentPPO'))


# === CONFIGURAÇÕES POR VARIÁVEIS DE AMBIENTE ===
USE_LSTM = os.getenv("USE_LSTM", "0") == "1"
ACTIVATION = nn.ReLU if os.getenv("ACTIVATION", "relu") == "relu" else nn.Tanh
USE_PENALTIES = os.getenv("USE_PENALTIES", "1") == "1"
RANDOM_OBSTACLES = os.getenv("RANDOM_OBSTACLES", "1") == "1"
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 10000))

# === IMPORTAÇÃO DO AMBIENTE ===
if USE_LSTM:
    from p2_thymio_rl_controller_RecurrentPPO import OpenAIGymEnvironment
    model_class = RecurrentPPO
    policy = 'MlpLstmPolicy'
else:
    from p2_thymio_rl_controller_PPO import OpenAIGymEnvironment
    model_class = PPO
    policy = 'MlpPolicy'

# === NOMES E LOGGING ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f"{'R' if USE_LSTM else ''}PPO_{'ReLU' if ACTIVATION==nn.ReLU else 'Tanh'}_" \
         f"{'randObs' if RANDOM_OBSTACLES else 'fixedObs'}_" \
         f"{'penalty' if USE_PENALTIES else 'noPenalty'}"

checkpoint_path = f"./checkpoints/{suffix}_{timestamp}"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs("logs_treino", exist_ok=True)
log_filename = f"logs_treino/{suffix}_{timestamp}.log"

# === LOGGING ===
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

def env_fn():
    return OpenAIGymEnvironment(
        reward_config=reward_config,
        random_obstacles=RANDOM_OBSTACLES
    )

env = make_vec_env(env_fn)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env.reset()

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
    # Criar pasta de modelos
    models_path = os.path.join("models")
    os.makedirs(models_path, exist_ok=True)

    model_filename = f"{suffix}_{timestamp}_model"
    vecnorm_filename = f"{suffix}_{timestamp}_vecnormalize.pkl"

    model.save(os.path.join(models_path, model_filename))
    env.save(os.path.join(models_path, vecnorm_filename))

    print("Modelo treinado e guardado com sucesso.")

    Supervisor().simulationQuit(0)

except KeyboardInterrupt:
    print("Treino interrompido manualmente. A guardar modelo...")
    model.save(f"{suffix}_{timestamp}_interrupt")
