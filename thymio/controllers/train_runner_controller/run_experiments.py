import subprocess
from itertools import product
import os
import time

# COMBINAÇÕES DE PARÂMETROS
algorithms = ['ppo', 'recurrent_ppo']
activations = ['relu', 'tanh']
penalties = [True, False]
random_obstacles = [True, False]

# CONFIGURAÇÕES
webots_exec = r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
world_path = r"C:\Users\Tiago\Documents\GitHub\IAR2025\thymio\worlds\projeto2.wbt"
total_timesteps = '150000'

# GERAR TODAS AS COMBINAÇÕES
combinations = list(product(algorithms, activations, penalties, random_obstacles))

for algo, act, penalty, randobs in combinations:
    env_vars = os.environ.copy()
    env_vars["USE_LSTM"] = "1" if algo == "recurrent_ppo" else "0"
    env_vars["ACTIVATION"] = act
    env_vars["USE_PENALTIES"] = "1" if penalty else "0"
    env_vars["RANDOM_OBSTACLES"] = "1" if randobs else "0"
    env_vars["TOTAL_TIMESTEPS"] = total_timesteps

    tag = f"{algo}_{act}_{'penalty' if penalty else 'noPenalty'}_{'randObs' if randobs else 'fixedObs'}"
    print(f"\n🔁 A correr treino: {tag}")

    # Lançar Webots
    proc = subprocess.Popen([
        webots_exec,
        "--mode=fast",
        "--stdout", "--stderr",
        world_path
    ], env=env_vars)

    # Esperar o fim do processo (bloqueia até o Webots fechar)
    proc.wait()

    # Pausa breve entre treinos para evitar conflitos de portas
    time.sleep(5)
