import numpy as np
import gzip
import shutil
import matplotlib.pyplot as plt
import time
import pandas as pd
import networkx as nx
import tsplib95

# Function to extract .gz file
def extract_gzip(source_filepath, destination_filepath):
    with gzip.open(source_filepath, 'rb') as f_in, open(destination_filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Function to load coordinates from the .tsp file
def load_coordinates(file_path):
    coordinates = []
    start_reading = False
    with open(file_path, 'r') as file:
        for line in file:
            if "NODE_COORD_SECTION" in line:
                start_reading = True
            elif "EOF" in line:
                break
            elif start_reading:
                parts = line.strip().split()
                if len(parts) == 3 and parts[0].isdigit():
                    coordinates.append((float(parts[1]), float(parts[2])))
    return coordinates

# Function to calculate fluxes
def calculate_fluxes(pressures, conductivities, distances, epsilon=1e-10):
    num_cities = len(pressures)
    fluxes = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance = max(distances[i][j], epsilon)  # Avoid division by zero
                delta_p = np.clip(pressures[i] - pressures[j], -1e3, 1e3)  # Ensure pressures do not cause overflow
                fluxes[i][j] = (conductivities[i][j] / distance) * delta_p
    return fluxes

# Function to update conductivities
def update_conductivities(conductivities, fluxes, decay_rate, k, m, scale_factor):
    num_cities = len(conductivities)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                flux = np.clip(abs(fluxes[i][j]), 0, 1e3)  # Control the size of the flux to prevent overflow
                growth = k * (flux ** m) - decay_rate * conductivities[i][j]
                growth = np.clip(growth, -1e3, 1e3)  # Limit the growth to prevent explosive updates
                conductivities[i][j] += growth + np.random.rand() * growth * scale_factor # Add randomness to explore more paths
    return conductivities

# Function to update pressures
def update_pressures(pressures, fluxes, rate=0.05):
    num_cities = len(pressures)
    for i in range(num_cities):
        net_influx = np.sum(fluxes[:, i]) - np.sum(fluxes[i, :])
        pressures[i] += rate * net_influx
        pressures[i] = np.clip(pressures[i], 0, 1e2)  # Prevent pressure from becoming negative or excessively high
    return pressures

# Function to prune connections
def prune_connections(conductivities, threshold=0.005):
    conductivities[conductivities < threshold] = 0
    return conductivities

# Function to construct TSP path
def construct_tsp_path(conductivities, fluxes):
    num_cities = len(conductivities)
    visited = np.zeros(num_cities, dtype=bool)
    path = []

    start_city = np.random.randint(0, num_cities) # made start_city random
    path.append(start_city)
    visited[start_city] = True

    current_city = start_city

    while len(path) < num_cities:
        next_city = -1
        max_value = 0 # if conductivity is zero the value will be zero we prune these connections so we are never interested in them

        for j in range(num_cities):
            if not visited[j]:
                value = conductivities[current_city][j] * np.abs(fluxes[current_city][j])
                if value > max_value:
                    next_city = j
                    max_value = value

        if next_city == -1:
            return None # instead of breaking return None so also the iteration can continue

        path.append(next_city)
        visited[next_city] = True
        current_city = next_city

    if len(path) == num_cities:
      path.append(start_city)
      return path

    return None # same logic as the breaking change

# Function to calculate the total distance of a given path
def calculate_total_distance(path, distances):
    total_distance = 0
    for i in range(len(path) - 1):
        if distances[path[i], path[i + 1]] == np.inf:
            print(f"Invalid distance between {path[i]} and {path[i + 1]}")
            return np.inf
        total_distance += distances[path[i], path[i + 1]]
    return total_distance

# Function to calculate memory usage
def calculate_memory_usage(conductivities):
    return conductivities.nbytes

# Function to plot the best path
def plot_best_path(best_path, cities, ax=None):
    path_coords = cities[best_path]

    if ax is None:
        ax = plt.gca()
    else:
        ax.clear()

    ax.plot(path_coords[:, 0], path_coords[:, 1], 'o-', markersize=10)
    ax.plot([path_coords[-1, 0], path_coords[0, 0]], [path_coords[-1, 1], path_coords[0, 1]], 'o-', markersize=10)  # Connect last to first

    for i, coord in enumerate(path_coords):
        ax.annotate(str(best_path[i]), (coord[0], coord[1]))  # Use index from best_path for annotation

    ax.set_title("Best Path")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)

# Function to plot the fitness over iterations
def plot_fitness_over_iterations(all_fitness, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.clear()
    ax.plot(all_fitness, 'o-', markersize=5)
    ax.set_title("Fitness over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Distance (Fitness)")
    ax.grid(True)

# Simulation loop function
def run_simulation(dataset_name, decay_rate, k, m, epsilon):
    # Extract and load city coordinates
    source_filepath = f'{dataset_name}.tsp.gz'
    destination_filepath = f'{dataset_name}.tsp'
    extract_gzip(source_filepath, destination_filepath)
    cities = np.array(load_coordinates(destination_filepath))
    print(f"Loaded coordinates for {len(cities)} cities from {dataset_name}.")

    with open(destination_filepath) as f:
        text = f.read()

    problem = tsplib95.parse(text)
    full_matrix = nx.adjacency_matrix(problem.get_graph()).todense()
    distances = np.array(full_matrix)
    distances = distances.astype(float)
    distances[distances == 1] = np.inf

    # Initialize pressures and conductivities
    pressures = np.random.rand(len(cities)) * 10
    conductivities = np.ones_like(distances) * 0.01  # Initial small value for all conductivities

    num_iterations = 100
    best_fitness = np.inf
    best_path = None
    best_distances = []
    computation_times = []
    convergence_rates = []
    memory_usages = []
    all_solutions = []
    all_paths = []

    start_time = time.time()

    for iteration in range(num_iterations):
        iteration_start_time = time.time()

        fluxes = calculate_fluxes(pressures, conductivities, distances, epsilon)
        pressures = update_pressures(pressures, fluxes)
        conductivities = update_conductivities(conductivities, fluxes, decay_rate, k, m)

        iteration_end_time = time.time()
        computation_times.append(iteration_end_time - iteration_start_time)

        conductivities = prune_connections(conductivities)
        current_path = construct_tsp_path(conductivities, fluxes)
        if current_path is None:
            continue

        current_fitness = calculate_total_distance(current_path, distances)
        all_paths.append(current_path)
        all_solutions.append(current_fitness)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_path = current_path

        best_distances.append(best_fitness)
        convergence_rates.append(best_fitness)
        memory_usages.append(calculate_memory_usage(conductivities))

        # Print the length of the tour for the current best path
        if best_path is not None:
            tour_length = calculate_total_distance(best_path, distances)
            #print(f"Iteration {iteration}, Best Path Length: {tour_length}")

    total_computation_time = time.time() - start_time

    print(f"Best Fitness for {dataset_name}:", best_fitness)
    print("Best Path:", best_path)
    print("Total Computation Time:", total_computation_time, "seconds")
    print("Average Computation Time per Iteration:", np.mean(computation_times), "seconds")
    print("Memory Usage:", memory_usages[-1], "bytes")

    # Plot the best path
    plt.figure()
    plot_best_path(best_path, cities)
    plt.title(f"Best Path for {dataset_name}")
    plt.show()

# Define the parameter sets
decay_rates = [0.001, 0.01, 0.1, 'auto']
ks = [0.01, 0.05, 0.1, 0.5]
ms = [1, 2, 3, 4]
epsilons = [1e-10, 1e-8, 1e-6]

# Fixed parameters
fixed_decay_rate = 0.01
fixed_k = 0.1
fixed_m = 2
fixed_epsilon = 1e-10

# Run the simulation with different parameter values
datasets = ['st70', 'ulysses22', 'ch150']
for dataset in datasets:
    for decay_rate in decay_rates:
        if decay_rate == 'auto':
            decay_rate = 1 / len(load_coordinates(f'{dataset}.tsp'))**2
        print(f"Running simulation with decay_rate={decay_rate}, k={fixed_k}, m={fixed_m}, epsilon={fixed_epsilon}")
        run_simulation(dataset, decay_rate, fixed_k, fixed_m, fixed_epsilon)
    for k in ks:
        print(f"Running simulation with decay_rate={fixed_decay_rate}, k={k}, m={fixed_m}, epsilon={fixed_epsilon}")
        run_simulation(dataset, fixed_decay_rate, k, fixed_m, fixed_epsilon)
    for m in ms:
        print(f"Running simulation with decay_rate={fixed_decay_rate}, k={fixed_k}, m={m}, epsilon={fixed_epsilon}")
        run_simulation(dataset, fixed_decay_rate, fixed_k, m, fixed_epsilon)
    for epsilon in epsilons:
        print(f"Running simulation with decay_rate={fixed_decay_rate}, k={fixed_k}, m={fixed_m}, epsilon={epsilon}")
        run_simulation(dataset, fixed_decay_rate, fixed_k, fixed_m, epsilon)
