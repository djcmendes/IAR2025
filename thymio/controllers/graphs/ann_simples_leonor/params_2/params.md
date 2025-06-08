# Parâmetros da Simulação
TIME_STEP = 64
POPULATION_SIZE = 30
PARENTS_KEEP = 4
INPUT = 2
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (INPUT * HIDDEN) + HIDDEN + (HIDDEN * OUTPUT) + OUTPUT
GENERATIONS = 300
EVALUATION_TIME = 200  # segundos
MAX_SPEED = 6.28
BASE_SPEED = 3.0

# --- PARÂMETROS DE MUTAÇÃO ADAPTATIVA ---
# Começa com valores altos para incentivar a exploração
INITIAL_MUTATION_RATE = 0.25
INITIAL_MUTATION_SIZE = 0.5
# Termina com valores baixos para permitir o refinamento
FINAL_MUTATION_RATE = 0.05
FINAL_MUTATION_SIZE = 0.1
# Percentagem de gerações durante a qual a mutação diminui
MUTATION_DECAY_FACTOR = 0.75 
MUTATION_DECAY_GENERATIONS = int(GENERATIONS * MUTATION_DECAY_FACTOR)