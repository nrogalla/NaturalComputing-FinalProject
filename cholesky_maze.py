import numpy as np
import gzip
import shutil
import matplotlib.pyplot as plt
import time
import pandas as pd
import networkx as nx
import tsplib95
import tracemalloc
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
from scipy.linalg import cholesky

def extract_gzip(source_filepath, destination_filepath):
    """
    Extracts a .gz file and saves it to the specified destination filepath.

    Args:
        source_filepath (str): The path to the source .gz file.
        destination_filepath (str): The path to save the extracted file.
    """
    with gzip.open(source_filepath, 'rb') as f_in, open(destination_filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def load_coordinates(file_path):
    """
    Load coordinates from a .tsp file.

    Parameters:
        file_path (str): The path to the .tsp file.

    Returns:
        list: A list of tuples representing the coordinates. Each tuple contains two floats representing the x and y coordinates respectively.
    """
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

def calculate_fluxes(pressures, conductivities, distances, epsilon=1e-10):
    """
    Calculate the fluxes between cities based on pressures, conductivities, and distances.

    Args:
        pressures (ndarray): An array of pressures for each city.
        conductivities (ndarray): An array of conductivities for each pair of cities.
        distances (ndarray): An array of distances between each pair of cities.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
        ndarray: An array of fluxes between each pair of cities.
    """
    num_cities = len(pressures)
    fluxes = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance = max(distances[i][j], epsilon)  # Avoid division by zero
                delta_p = np.clip(pressures[i] - pressures[j], -1e3, 1e3)  # Ensure pressures do not cause overflow
                fluxes[i][j] = (conductivities[i][j] / distance) * delta_p
    return fluxes

def type_i_function(flux, k, m):
    """
    Type I function (power function) for flux-conductance relationship.

    Parameters:
        flux (float): The flux value.
        k (float): The scaling factor.
        m (float): The exponent value.

    Returns:
        float: The result of the power function operation.
    """
    return k * (flux ** m)

def type_ii_function(flux, k, midpoint, steepness):
    """
    Type II function (sigmoidal profile) for flux-conductance relationship.

    Parameters:
        flux (float): The flux value.
        k (float): The scaling factor.
        midpoint (float): The midpoint value.
        steepness (float): The steepness value.

    Returns:
        float: The result of the sigmoidal profile function operation.
    """
    return k / (1 + np.exp(-steepness * (flux - midpoint)))

def solve_poisson_iccg(pressures, fluxes, conductivities, distances, epsilon=1e-10):
    """
    Function to solve the Poisson equation using ICCG method.

    Parameters:
        pressures (ndarray): An array of pressures for each city.
        fluxes (ndarray): An array of fluxes for each city pair.
        conductivities (ndarray): An array of conductivities for each city pair.
        distances (ndarray): An array of distances between each city pair.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
        ndarray: An array of updated pressures after solving the Poisson equation.
    """
    num_cities = len(pressures)
    A = np.zeros((num_cities, num_cities))
    b = np.zeros(num_cities)

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                A[i, i] += conductivities[i, j] / max(distances[i, j], epsilon)
                A[i, j] -= conductivities[i, j] / max(distances[i, j], epsilon)
        b[i] = np.sum(fluxes[:, i]) - np.sum(fluxes[i, :])

    A_sparse = csr_matrix(A)
    L = cholesky(A_sparse.todense() + np.eye(num_cities) * epsilon, lower=True)
    pressures, _ = cg(A_sparse, b, x0=pressures, M=L)
    return pressures

def update_conductivities(conductivities, fluxes, decay_rate, k, m, noise_factor, func_type='type_i', midpoint=None, steepness=None):
    """
    Updates the conductivities based on the given fluxes, decay rate, and function type.

    Args:
        conductivities (ndarray): The conductivities matrix.
        fluxes (ndarray): The flux matrix.
        decay_rate (float): The decay rate.
        k (float): The scaling factor for the type_i function.
        m (float): The exponent value for the type_i function.
        noise_factor (float): The noise factor.
        func_type (str, optional): The function type. Defaults to 'type_i'.
        midpoint (float, optional): The midpoint value for the type_ii function.
        steepness (float, optional): The steepness value for the type_ii function.

    Returns:
        ndarray: The updated conductivities matrix.

    Raises:
        ValueError: If the function type is invalid or midpoint and steepness are not provided for type_ii function.

    """
    num_cities = len(conductivities)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                flux = np.clip(abs(fluxes[i][j]), 0, 1e3)  # Control the size of the flux to prevent overflow
                if func_type == 'type_i':
                    growth = type_i_function(flux, k, m)
                elif func_type == 'type_ii':
                    if midpoint is None or steepness is None:
                        raise ValueError("midpoint and steepness must be provided for type_ii function")
                    growth = type_ii_function(flux, k, midpoint, steepness)
                else:
                    raise ValueError("Invalid function type. Use 'type_i' or 'type_ii'.")

                # Semi-implicit update
                conductivities[i][j] = (conductivities[i][j] + growth * noise_factor) / (1 + decay_rate)
                conductivities[i][j] = np.clip(conductivities[i][j], -1e3, 1e3)  # Limit the conductivities to prevent overflow
    return conductivities

def update_pressures(pressures, fluxes, rate=0.05):
    """
    Update the pressures based on the fluxes.

    Args:
        pressures (ndarray): An array of pressures for each city.
        fluxes (ndarray): An array of fluxes between each pair of cities.
        rate (float, optional): The rate at which the pressures are updated. Defaults to 0.05.

    Returns:
        ndarray: The updated array of pressures.
    """
    num_cities = len(pressures)
    for i in range(num_cities):
        net_influx = np.sum(fluxes[:, i]) - np.sum(fluxes[i, :])
        pressures[i] += rate * net_influx
        pressures[i] = np.clip(pressures[i], 0, 1e2)  # Prevent pressure from becoming negative or excessively high
    return pressures

def prune_connections(conductivities, threshold=0.005):
    """
    Prune connections in the conductivities matrix based on a given threshold.

    Args:
        conductivities (ndarray): The conductivities matrix.
        threshold (float, optional): The threshold value below which connections are pruned. Defaults to 0.005.

    Returns:
        ndarray: The updated conductivities matrix with connections pruned.
    """
    conductivities[conductivities < threshold] = 0
    return conductivities

def construct_tsp_path(conductivities, fluxes):
    """
    Construct a TSP (Traveling Salesman Problem) path using the given conductivities and fluxes.

    Parameters:
        conductivities (ndarray): A 2D array representing the conductivities between each pair of cities.
        fluxes (ndarray): A 2D array representing the fluxes between each pair of cities.

    Returns:
        list: A list representing the TSP path. If no path can be constructed, returns None.
    """
    num_cities = len(conductivities)
    visited = np.zeros(num_cities, dtype=bool)
    path = []

    start_city = np.random.randint(0, num_cities)  # Random starting city
    path.append(start_city)
    visited[start_city] = True

    current_city = start_city

    while len(path) < num_cities:
        next_city = -1
        max_value = 0

        for j in range(num_cities):
            if not visited[j]:
                value = conductivities[current_city][j] * np.abs(fluxes[current_city][j])
                if value > max_value:
                    next_city = j
                    max_value = value

        if next_city == -1:
            return None  # If no valid next city is found, return None

        path.append(next_city)
        visited[next_city] = True
        current_city = next_city

    if len(path) == num_cities:
        return path

    return None

def calculate_total_distance(path, distances):
    """
    Calculate the total distance of a given path using the given distances matrix.

    Parameters:
        path (list): A list of integers representing the path.
        distances (numpy.ndarray): A 2D numpy array representing the distances between each pair of cities.

    Returns:
        float: The total distance of the given path. If there is an invalid distance between two cities, returns np.inf.
    """    
    total_distance = 0
    for i in range(len(path) - 1):
        if distances[path[i], path[i + 1]] == np.inf:
            print(f"Invalid distance between {path[i]} and {path[i + 1]}")
            return np.inf
        total_distance += distances[path[i], path[i + 1]]

    total_distance += distances[path[-1], path[0]]
    return total_distance

def plot_best_path(best_path, cities, ax=None):
    """
    Plot the best path on a graph using the given best path, cities coordinates, and optional axes.

    Parameters:
        best_path (list): A list of integers representing the best path.
        cities (numpy.ndarray): A 2D numpy array representing the coordinates of the cities.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): The axes to plot on. If not provided, a new set of axes will be created. Defaults to None.
    """    
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

def plot_fitness_over_iterations(all_fitness, ax=None):
    """
    Plot the fitness over iterations.

    Args:
        all_fitness (list): A list of fitness values for each iteration.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): The axes to plot on. If not provided, a new set of axes will be created. Defaults to None.
    """    
    if ax is None:
        ax = plt.gca()
    ax.clear()
    ax.plot(all_fitness, 'o-', markersize=5)
    ax.set_title("Fitness over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Distance (Fitness)")
    ax.grid(True)

def run_simulation(dataset_name, noise_factor, func_type='type_i', midpoint=None, steepness=None):
    """
    Run the simulation for a given dataset with noise and function type.
    
    Args:
        dataset_name (str): The name of the dataset.
        noise_factor (float): The noise factor for the simulation.
        func_type (str, optional): Type of the function. Defaults to 'type_i'.
        midpoint (float, optional): The midpoint value. Defaults to None.
        steepness (float, optional): The steepness value. Defaults to None.
    
    Returns:
        Tuple: A tuple containing the best fitness value, total computation time, and total memory allocated.
    """
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

    decay_rate = 0.01
    k = 0.1
    m = 2
    epsilon = 1e-10

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
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    for _ in range(num_iterations):
        fluxes = calculate_fluxes(pressures, conductivities, distances)

        # Update pressures using ICCG
        pressures = solve_poisson_iccg(pressures, fluxes, conductivities, distances, epsilon)

        # Semi-implicit update of conductivities
        conductivities = update_conductivities(conductivities, fluxes, decay_rate, k, m, noise_factor, func_type, midpoint, steepness)

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

    total_computation_time = time.time() - start_time
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats_initial = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory_allocated_initial = sum(stat.size_diff for stat in stats_initial)

    print(f"Best Fitness for {dataset_name} with noise factor {noise_factor}:", best_fitness)
    print("Best Path:", best_path)
    print("Total Computation Time:", total_computation_time, "seconds")
    print("Average Computation Time per Iteration:", np.mean(computation_times), "seconds")
    print("Memory Usage:", memory_usages[-1], "bytes")
    print("Memory Usage: ", total_memory_allocated_initial/(1024**2), 'megabytes')

    # Plot the best path
    plt.figure()
    plot_best_path(best_path, cities)
    plt.title(f"Best Path for {dataset_name} (Noise factor {noise_factor})")
    plt.show()

    return best_fitness, total_computation_time, total_memory_allocated_initial