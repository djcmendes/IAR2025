#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Carregar os dados
df = pd.read_csv('./data/fitness_history_ANN_advanced_4n.csv')

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

# Adicionar linha de regressão linear
X = np.array(generations).reshape(-1, 1)
y = np.array(fitness)
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
plt.plot(generations, y_pred, color='red', linestyle='--', linewidth=2, label='Linear Regression')

# Legenda e layout
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./data/fitness_ann_adv_4n_plot.png', dpi=150)
plt.show()

## collisions

collisions = df['AverageActualCollisions'].tolist()

# Criar figura
plt.figure(figsize=(14, 7))

# Plot das colisões
plt.plot(generations, collisions, 'g-', linewidth=1.5, alpha=0.7, label='AverageActualCollisions')
plt.plot(generations, collisions, 'o', markersize=5, color='green', alpha=0.7)

# Eixos e título
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Average Collisions", fontsize=14, color='green')
plt.title("Collision Evolution Over Generations", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(np.arange(0, max(generations) + 1, 2), rotation=45, ha='right', fontsize=10)

# Anotação do valor máximo
max_collisions = max(collisions)
max_collisions_gen = generations[collisions.index(max_collisions)]
plt.annotate(
     f'Max Collisions: {max_collisions:.2f}',
    xy=(max_collisions_gen, max_collisions),
    xytext=(max_collisions_gen + 0.5, max_collisions + 0.1),
    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
)

# Adicionar linha de regressão linear
X = np.array(generations).reshape(-1, 1)
y = np.array(collisions)
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
plt.plot(generations, y_pred, color='red', linestyle='--', linewidth=2, label='Linear Regression')

# Legenda e layout
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./data/fitness_ann_adv_plot_4n_collisions.png', dpi=150)
plt.show()

