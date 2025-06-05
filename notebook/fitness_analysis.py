
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the dataset
df = pd.read_csv('fitness_history_ann_advanced_1.csv')

# Convert column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Calculate the average fitness score per generation
average_fitness_per_generation = df.groupby('generation')['fitness'].mean()

# Plot the average fitness score per generation
plt.figure(figsize=(10, 6))
average_fitness_per_generation.plot()
plt.title('Average Fitness Score per Generation')
plt.xlabel('Generation')
plt.ylabel('Average Fitness Score')
plt.grid(True)
plt.show()

# Clean and parse the 'genome' column using regex to extract float numbers
df['genome'] = df['genome'].apply(lambda x: [float(num) for num in re.findall(r'-?\d+\.\d+', x)])

# Select a few weights to visualize
selected_weights = [0, 1, 2, 3, 4]

# Plot the evolution of selected weights across generations
plt.figure(figsize=(10, 6))
for weight in selected_weights:
    plt.plot(df['generation'], df['genome'].apply(lambda x: x[weight] if len(x) > weight else None), label=f'Weight {weight}')

plt.title('Evolution of Selected Weights Across Generations')
plt.xlabel('Generation')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)
plt.show()
