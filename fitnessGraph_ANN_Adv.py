#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
df = pd.read_csv('./data/fitness_history_ANN_advanced.csv')

# Extrair colunas
generations = df['Generation'].tolist()
fitness = df['AverageFitness'].tolist()

# Criar figura
plt.figure(figsize=(14, 7))

# Plot do fitness
plt.plot(generations, fitness, 'b-', linewidth=1.5, alpha=0.7, label='AverageFitness')
plt.plot(generations, fitness, 'o', markersize=5, color='blue', alpha=0.7)

# Eixos e título
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Average Fitness", fontsize=14, color='blue')
plt.title("Fitness Evolution Over Generations", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(np.arange(0, max(generations) + 1, 2), rotation=45, ha='right', fontsize=10)

# Anotação do valor máximo
max_fitness = max(fitness)
max_fitness_gen = generations[fitness.index(max_fitness)]
plt.annotate(
    f'Max Fitness: {max_fitness:.2f}',
    xy=(max_fitness_gen, max_fitness),
    xytext=(max_fitness_gen + 0.5, max_fitness + 0.1),
    arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
)

# Legenda e layout
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./data/fitness_ann_adv_plot.png', dpi=150)
plt.show()