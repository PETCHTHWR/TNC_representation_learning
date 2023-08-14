import numpy as np

# Define the population (can be larger or smaller)
population = np.random.randn(10, 6, 2000)  # Replace with your actual data

# Define the desired sample size
sample_size = 40

# Sample with replacement
sample_indices = np.random.choice(population.shape[0], size=sample_size, replace=True)

# Extract the sampled data
sampled_data = population[sample_indices]

print("Sampled data shape:", sampled_data.shape)