#!/usr/bin/env python3

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <fitness_history_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    generations = []
    fitness = []
    
    try:
        with open(input_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                generations.append(int(row[0]))
                fitness.append(float(row[1]))
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
    
    if not generations:
        print("Error: No valid data found in the file.")
        sys.exit(1)
    
    plt.figure(figsize=(14, 7))
    
    # Create the plot with better spaced markers
    plt.plot(generations, fitness, 'b-', linewidth=1.5, alpha=0.7)  # Solid line
    plt.plot(generations, fitness, 'o', markersize=5, color='blue', alpha=0.7)  # Separate markers
    
    plt.title("Fitness Evolution Over Generations", fontsize=16)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Average Fitness", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Improved x-axis labeling
    max_gen = max(generations)
    plt.xticks(
        np.arange(0, max_gen + 1, 5 if max_gen > 30 else 2),  # Auto-adjust based on generation count
        rotation=45,
        ha='right',
        fontsize=10
    )
    plt.yticks(fontsize=10)
    
    # Set axis limits with padding
    plt.xlim(min(generations) - 0.5, max(generations) + 0.5)
    plt.ylim(min(fitness) * 0.9, max(fitness) * 1.1)
    
    # Highlight max fitness point with annotation
    max_fitness = max(fitness)
    max_gen = generations[fitness.index(max_fitness)]
    plt.annotate(f'Max: {max_fitness:.2f}',
                xy=(max_gen, max_fitness),
                xytext=(max_gen + 0.5, max_fitness + (max_fitness * 0.1)),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    plt.tight_layout()
    output_file = input_file.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=150)
    print(f"Generated fitness plot: {output_file}")

if __name__ == "__main__":
    main()