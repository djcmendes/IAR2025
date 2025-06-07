import os
import sys
from datetime import datetime
import logging
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# 🛠️ Define se queres usar PPO ou RecurrentPPO:
USE_LSTM = True  # True = RecurrentPPO

# 🛠️ Corrige o caminho para o import dependendo da escolha
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

ACTIVATION = nn.ReLU  # nn.Tanh também é possível
RANDOM_OBSTACLES = True
USE_PENALTIES = True
TOTAL_TIMESTEPS = 100000

# 📁 Preparar pasta de checkpoints e nome base
suffix = f"{'R' if USE_LSTM else ''}PPO_{'ReLU' if ACTIVATION==nn.ReLU else 'Tanh'}_{'randObs' if RANDOM_OBSTACLES else 'fixedObs'}_{'penalty' if USE_PENALTIES else 'noPenalty'}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = f"./checkpoints/{suffix}_{timestamp}"
os.makedirs(checkpoint_path, exist_ok=True)

# 🗂️ Configurar logging para ficheiro e consola
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/{suffix}_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# 📁 Preparar pasta de checkpoints
suffix = f"{'R' if USE_LSTM else 'P'}PO_{'ReLU' if ACTIVATION==nn.ReLU else 'Tanh'}_{'randObs' if RANDOM_OBSTACLES else 'fixedObs'}_{'penalty' if USE_PENALTIES else 'noPenalty'}"
checkpoint_path = f"./checkpoints/{suffix}_{timestamp}"
os.makedirs(checkpoint_path, exist_ok=True)

# Configurar função de recompensa
reward_config = {
    "penaliza_queda": USE_PENALTIES,
    "penaliza_proximidade": USE_PENALTIES,
    "recompensa_movimento": True,
    "recompensa_base": True
}

# Função para criar ambiente
def env_fn():
    return OpenAIGymEnvironment(
        reward_config=reward_config,
        random_obstacles=RANDOM_OBSTACLES
    )

# Criar ambiente vectorizado e normalizado
# env = DummyVecEnv([env_fn])
env = make_vec_env(env_fn)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

env.reset()

# Definir arquitetura e ativação
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

# Callback para guardar checkpoints
checkpoint = CheckpointCallback(save_freq=10000, save_path=checkpoint_path, name_prefix='thymio')

print(f"🔧 A treinar modelo: {suffix}")

try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint])
    model.save(f"{suffix}_model")
    env.save(f"{suffix}_vecnormalize")
    print("✅ Modelo treinado e guardado com sucesso.")

except KeyboardInterrupt:
    print("⚠️ Treino interrompido. A guardar modelo...")
    model.save(f"{suffix}_interrupt")
 

 # 