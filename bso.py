import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tsplib95
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random
import pandas as pd
import tracemalloc
import time
import itertools

MAX_DISTANCE = float('inf')

def get_pos(file_path, show_graph=False):
    """
    Get the position dictionary of the cities in the given TSP file.

    Parameters:
        file_path (str): The path to the TSP file.
        show_graph (bool, optional): Whether to show the graph representation of the TSP problem. Defaults to False.

    Returns:
        dict: A dictionary mapping the index of each city to its (x, y) coordinates.
    """
    # Open the file and read its contents
    with open(file_path) as f:
        text = f.read()
    problem = tsplib95.parse(text)

    # Get the graph representation of the TSP problem
    full_matrix = nx.adjacency_matrix(problem.get_graph()).todense()
    graph = np.array(full_matrix)

    # If the TSP problem has node coordinates, use them
    if not problem.node_coords:
        cities = pd.DataFrame(problem.display_data).T
    else:
        cities = pd.DataFrame(problem.node_coords).T

    cities.rename(columns={0: 'x', 1: 'y'}, inplace=True)
    cities = cities.astype(int)
    position_dict = {index-1: (row['x'], row['y']) for index, row in cities.iterrows()}

    return position_dict

def generate_random_individual(num_cities):
    """
    Generates a random individual for a Traveling Salesman Problem (TSP) by shuffling the indices of the cities.

    Parameters:
        num_cities (int): The number of cities in the TSP.

    Returns:
        numpy.ndarray: A random individual representing a tour in the TSP.
    """
    individual = np.array(np.arange(num_cities))
    np.random.shuffle(individual)
    return individual

def calculate_fitness(individual, distances):
    """
    Calculate the total distance of a given individual in the Traveling Salesman Problem (TSP).

    Parameters:
        individual (numpy.ndarray): An array representing the individual in the TSP.
        distances (numpy.ndarray): A matrix of distances between cities.

    Returns:
        float: The total distance of the individual.
    """
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distances[individual[i]][individual[i+1]]
    total_distance += distances[individual[-1]][individual[0]]  # Return to the starting city
    return total_distance

def calculate_fitness_with_penalty(individual, distances):
    """
    Calculate the fitness of an individual in the Traveling Salesman Problem (TSP) with a penalty for not visiting all cities.

    Parameters:
        individual (numpy.ndarray): The individual representing a tour in the TSP.
        distances (numpy.ndarray): The matrix of distances between cities.

    Returns:
        float: The fitness value of the individual, including a penalty for not visiting all cities.
            If all cities are visited, the fitness value is the total distance of the tour.
            Otherwise, the fitness value is set to a maximum distance value (MAX_DISTANCE).

    """
    total_distance = 0
    visited_cities = set()
    for i in range(len(individual) - 1):
        total_distance += distances[individual[i]][individual[i+1]]
        visited_cities.add(individual[i])
    total_distance += distances[individual[-1]][individual[0]]  # Return to the starting city
    visited_cities.add(individual[-1])

    # Penalty for not visiting all cities
    if len(distances) != len(visited_cities):
      return MAX_DISTANCE

    return total_distance

def kmeans_clustering_with_highest_fitness(individuals, k, distances, p_clustering):
    """
    Perform K-means clustering on the given individuals and calculate the highest fitness within each cluster.
    
    Parameters:
        individuals (numpy.ndarray): The array of individuals representing tours in the TSP.
        k (int): The number of clusters.
        distances (numpy.ndarray): The matrix of distances between cities.
        p_clustering (float): The probability of randomly replacing a clustering center.
    
    Returns:
        tuple: A tuple containing two elements:
            - individuals_by_cluster (list): A list of lists, where each inner list represents a cluster and contains tuples of individuals and their indices.
            - new_centers (list): A list of tuples, where each tuple represents a new center and contains an individual and its index.
    """
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(individuals)

    individuals_by_cluster = [[] for _ in range(k)]
    i= 0
    for individual, label in zip(individuals, kmeans.labels_):
        individuals_by_cluster[label].append((individual, i))
        i+= 1
    
    # The individual with the highest fitness within the cluster will be the center cluster
    new_centers = []

    for cluster_individuals in individuals_by_cluster:
        if cluster_individuals:
            cluster_fitness = [calculate_fitness_with_penalty(individual[0], distances) for individual in cluster_individuals]
            max_fitness_idx = np.argmin(cluster_fitness)
            new_centers.append(cluster_individuals[max_fitness_idx])
        else:
            new_centers.append(individuals[np.random.choice(len(individuals))])  # If the cluster is empty, choose a random individual
    
    # Random replacing of clustering center
    r_clustering = random.random()
    if r_clustering < p_clustering:
      chosen_cluster = np.random.choice(len(new_centers))
      new_indiv_index = np.random.choice(len(individuals))
      new_centers[chosen_cluster] = (individuals[new_indiv_index], new_indiv_index)

    return individuals_by_cluster, new_centers

def new_individual_generation(individuals_by_cluster, kmeans_centers, p_gen, p_one_cluster, p_two_cluster, distances):
    """
    Generate a new individual based on the given parameters.

    Args:
        individuals_by_cluster (dict): A dictionary mapping cluster labels to lists of individuals.
        kmeans_centers (list): A list of cluster centers.
        p_gen (float): The probability of generating a new individual.
        p_one_cluster (float): The probability of selecting a cluster center as the new individual.
        p_two_cluster (float): The probability of performing cluster center crossover.
        distances (numpy.ndarray): The distance matrix.

    Returns:
        tuple: A tuple containing the new individual and its corresponding cluster label. If the new individual is not generated, returns (None, None).

    """
    r_gen  = random.random()

    # One Cluster Case
    if r_gen < p_gen:
      i = np.random.choice(len(kmeans_centers))
      r_one_cluster = random.random()
      # Cluster center
      if r_one_cluster < p_one_cluster:

          X_select = kmeans_centers[i]
      # normal individual
      else:
          j = np.random.choice(len(individuals_by_cluster[i]))

          X_select = individuals_by_cluster[i][j]

      X_select = (invert_part(X_select[0]), X_select[1])
      X_select_2 = None

    # Two Cluster Case
    else:
        r_two_cluster = random.random()
        # Randomly select two clusters
        i = np.random.choice(len(kmeans_centers))
        while True:
            j = np.random.choice(len(kmeans_centers))
            if i != j:
              break
        # cluster center crossover
        if r_two_cluster < p_two_cluster:

          # child_1, child_2 = order_crossover(kmeans_centers[i][0], kmeans_centers[j][0])
          child_1 = heuristic_crossover(kmeans_centers[i][0], kmeans_centers[j][0], distances)
          child_2 = heuristic_crossover(kmeans_centers[i][0], kmeans_centers[j][0], distances)
          X_select = (child_1, kmeans_centers[i][1])
          X_select_2 = (child_2, kmeans_centers[j][1])

        # normal individual crossover
        else:
          ind1 = np.random.choice(len(individuals_by_cluster[i]))
          ind2 = np.random.choice(len(individuals_by_cluster[j]))
          child_1 = heuristic_crossover(individuals_by_cluster[i][ind1][0], individuals_by_cluster[j][ind2][0], distances)
          child_2 = heuristic_crossover(individuals_by_cluster[i][ind1][0], individuals_by_cluster[j][ind2][0], distances)
          X_select = (child_1, individuals_by_cluster[i][ind1][1])
          X_select_2 = (child_2, individuals_by_cluster[j][ind2][1])

    return X_select, X_select_2

def tsp_bso(file_path, n_clusters, num_individuals, max_iter, p_clustering, p_gen, p_one_cluster, p_two_cluster, optimal_fitness, get_iteration_fitness = False):
    """
    Perform a binary search optimization (BSO) algorithm for the Traveling Salesman Problem (TSP) using the given parameters.

    Args:
        file_path (str): The path to the TSP file.
        n_clusters (int): The number of clusters for clustering.
        num_individuals (int): The number of potential solutions to generate.
        max_iter (int): The maximum number of iterations.
        p_clustering (float): The probability of clustering.
        p_gen (float): The probability of generating a new individual.
        p_one_cluster (float): The probability of selecting a cluster center.
        p_two_cluster (float): The probability of selecting a normal individual.
        optimal_fitness (float): The known optimum for the TSP.
        get_iteration_fitness (bool, optional): Whether to return the fitness values per iteration. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - distances (numpy.ndarray): The distance matrix.
            - best_individual (list): The best individual found.
            - best_fitness_per_iteration (list): The fitness values per iteration if `get_iteration_fitness` is True.
            - gen (int): The number of iterations performed.

            If `get_iteration_fitness` is False, only the first three values are returned.
    """
    with open(file_path) as f:
      text = f.read()

    problem = tsplib95.parse(text)
    full_matrix = nx.adjacency_matrix(problem.get_graph()).todense()
    distances = np.array(full_matrix)
    num_cities = len(distances)

    # Randomly generate n potential solutions
    individuals = [generate_random_individual(num_cities) for _ in range(num_individuals)]
    best_fitness = float('inf')
    best_individual = None
    gen = 1
    best_fitness_per_iteration = []

    while gen <= max_iter and best_fitness > optimal_fitness: # and iter_without_improvement < max_iter_without_improvement:

        # cluster
        individuals_by_cluster, kmeans_centers = kmeans_clustering_with_highest_fitness(individuals, n_clusters, distances, p_clustering)

        # new individual generation
        X_selections = new_individual_generation(individuals_by_cluster, kmeans_centers, p_gen, p_one_cluster, p_two_cluster, distances)
        # print(X_selections)

        # Selection
        for X_select in X_selections:
          if X_select is not None:
            new_fitness = calculate_fitness_with_penalty(X_select[0], distances)
            if new_fitness < calculate_fitness_with_penalty(individuals[X_select[1]], distances):
                individuals[X_select[1]] = X_select[0]


            # Evaluation
            individual_fitness = calculate_fitness_with_penalty(individuals[X_select[1]], distances)
            if individual_fitness < best_fitness:
                #print(individual_fitness)
                best_fitness = individual_fitness
                best_individual = individuals[X_select[1]]

        gen += 1
        best_fitness_per_iteration.append(best_fitness)

    if get_iteration_fitness:
      return distances, best_individual, best_fitness_per_iteration, gen
    return distances, best_individual, best_fitness, gen

def invert_part(individual):
    """
    Invert a part of an individual by randomly selecting two distinct positions and swapping the values between them.

    Parameters:
        individual (list): The individual representing a TSP tour.

    Returns:
        list: The individual with the selected part inverted.
    """
    # Randomly select two distinct positions
    position1, position2 = random.sample(range(0, len(individual)), k=2)

    # Swap the values at the selected positions
    inverted_part = np.flip(individual[position1:position2])
    individual[position1:position2] = inverted_part

    return individual

def calculate_path_distance(path, distances):
    """
    Calculate the total distance of a path in a graph.

    Parameters:
        path (list): A list of nodes representing the path.
        distances (numpy.ndarray): A matrix of distances between nodes.

    Returns:
        float: The total distance of the path.
    """
    return sum(distances[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))

def heuristic_crossover(parent1, parent2, distances):
    """
    Perform heuristic crossover between two parents to generate a child path.

    Parameters:
        parent1 (list): The first parent path.
        parent2 (list): The second parent path.
        distances (numpy.ndarray): Matrix of distances between cities.

    Returns:
        list: The generated child path after crossover.
    """
    # Convert parent arrays to lists for easier manipulation
    parent1 = list(parent1)
    parent2 = list(parent2)

    # Randomly select a city m
    m = np.random.randint(0, len(parent1))

    # Find the index of m in both parents
    idx1 = parent1.index(m)
    idx2 = parent2.index(m)

    # Determine the right and left neighbors of m in both parents
    m_right1 = parent1[(idx1 + 1) % len(parent1)]
    m_left1 = parent1[(idx1 - 1) % len(parent1)]

    m_right2 = parent2[(idx2 + 1) % len(parent2)]
    m_left2 = parent2[(idx2 - 1) % len(parent2)]

    # List of candidate cities to add to the child
    candidates = [m_right1, m_right2, m_left1, m_left2]

    # Initialize the child with the starting city m
    child = [m]
    path_lengths = []

    # Continue adding cities to the child until it reaches the desired length
    while len(child) < len(parent1) + len(child) - 1:
        path_lengths.clear()

        # Evaluate each candidate city
        for i, candidate in enumerate(candidates):
            if candidate in child:
                path_lengths.append(np.inf)
                continue

            # Temporarily add the candidate to the child and calculate path distance
            if i < 2:
                child.append(candidate)
                path_lengths.append(calculate_path_distance(child, distances))
                child.pop()
            else:
                child.insert(0, candidate)
                path_lengths.append(calculate_path_distance(child, distances))
                child.pop(0)

        # Select the candidate with the minimum path distance
        c_idx = np.argmin(path_lengths)
        selected_candidate = candidates[c_idx]

        # Add the selected candidate to the child path
        if c_idx == 0 or c_idx == 1:
            child.append(selected_candidate)
        else:
            child.insert(0, selected_candidate)

        # Remove the selected candidate from both parents
        parent1 = [city for city in parent1 if city != selected_candidate]
        parent2 = [city for city in parent2 if city != selected_candidate]

        # Update the candidates list with new neighbors if parents are not empty
        if parent1 and parent2:
            idx1 = parent1.index(m)
            idx2 = parent2.index(m)

            m_right1 = parent1[(idx1 + 1) % len(parent1)]
            m_left1 = parent1[(idx1 - 1) % len(parent1)]

            m_right2 = parent2[(idx2 + 1) % len(parent2)]
            m_left2 = parent2[(idx2 - 1) % len(parent2)]

            candidates = [m_right1, m_right2, m_left1, m_left2]
        else:
            break

    return child

def get_graph_from_path(tsp_solution):
    """
    Creates a graph representation of the given TSP solution path.

    Args:
        tsp_solution (List[int]): A list of integers representing the TSP solution path.

    Returns:
        nx.Graph: A graph object representing the TSP solution path.

    """
    # Create a graph for the TSP solution path
    G = nx.Graph()

    # Add nodes and edges for the TSP solution
    edges = [(tsp_solution[i], tsp_solution[i + 1]) for i in range(len(tsp_solution) - 1)]
    edges.append((tsp_solution[0],tsp_solution[-1] ))
    G.add_edges_from(edges)
    return G

def plot_tsp_graph_with_solution(filepath, solution, best_fitness):
    """
    Plot the Traveling Salesman Problem (TSP) graph with the given solution path.

    Parameters:
        filepath (str): The path to the TSP file.
        solution (list): A list of cities in the order they should be visited.
        best_fitness (float): The fitness value representing the total distance of the solution path.

    Returns:
        None
    """
    pos = get_pos(filepath)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot each graph
    nx.draw(get_graph_from_path(solution), pos,
            node_color='C1',
            node_shape='s',
            node_size=12,
            with_labels=False,
            ax=ax)
    
    fig.text(0.5, 0.01, f'Distance: {best_fitness}', ha='center')
    plt.show()

def run_parameter_optimization(file_path, optimal_fitness = 0):
    """
    Run a parameter optimization for the Traveling Salesman Problem (TSP) using the given file path.

    Args:
        file_path (str): The path to the TSP file.
        optimal_fitness (float, optional): The known optimum for the TSP. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - all_parameters (list): A list of tuples representing the parameter values explored.
            - all_fitness_means (list): A list of floats representing the mean fitness values for each parameter set.
            - all_fitness_sds (list): A list of floats representing the standard deviation of fitness values for each parameter set.
    """
    with open(file_path) as f:
        text = f.read()

    problem = tsplib95.parse(text)
    full_matrix = nx.adjacency_matrix(problem.get_graph()).todense()
    graph = np.array(full_matrix)
    num_cities = len(graph)
    num_individuals = 100

    # Define parameter values to explore
    p_gen_values = [0.5, 0.65]
    p_one_cluster_values = [0.3, 0.4, 0.5]
    p_two_cluster_values = [0.4, 0.55, 0.7]
    p_clustering_values = [0.2, 0.3, 0.4]
    n_clusters = 5
    n_iter = 1000

    # Perform grid search
    best_fitness = float('inf')
    best_parameters = None
    all_parameters = []
    all_fitness_means = []
    all_fitness_sds = []

    i = 0
    for p_gen, p_one_cluster, p_two_cluster, p_clustering in itertools.product(p_gen_values, p_one_cluster_values, p_two_cluster_values, p_clustering_values):
        fitness_values = []
        i += 1
        print(f'Test: ', i)
        print(p_gen, p_one_cluster, p_two_cluster, p_clustering)
        for _ in range(10):
            graph, _, fitness, _ = tsp_bso(file_path, n_clusters, num_individuals, n_iter, p_clustering, p_gen, p_one_cluster, p_two_cluster, optimal_fitness, False)
            fitness_values.append(fitness)

        mean_fitness = np.mean(fitness_values)
        sd_fitness = np.std(fitness_values)

        parameters = (p_gen, p_one_cluster, p_two_cluster, p_clustering)

        all_parameters.append(parameters)
        all_fitness_means.append(mean_fitness)
        all_fitness_sds.append(sd_fitness)

    return all_parameters, all_fitness_means, all_fitness_sds

def run_tsp_with_stats(file_path, optimum, p_gen, p_one_cluster, p_two_cluster, p_clustering):
    """
    Run Traveling Salesman Problem (TSP) with statistics.

    Args:
        file_path (str): The path to the TSP file.
        optimum (float): The known optimum for the TSP.
        p_gen (float): The probability of generating a new individual.
        p_one_cluster (float): The probability of selecting an individual from one cluster.
        p_two_cluster (float): The probability of selecting an individual from two clusters.
        p_clustering (float): The probability of clustering the individuals.

    Returns:
        tuple: A tuple containing the best fitness value, elapsed time, and total memory allocated.
    """
    num_individuals = 100
    max_iter = 1000
    n_clusters = 5

    start_time = time.time()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    _, _, best_fitness, _ = tsp_bso(file_path, n_clusters, num_individuals, max_iter, p_clustering, p_gen, p_one_cluster, p_two_cluster, optimum, False)

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats_mem = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory_allocated = sum(stat.size_diff for stat in stats_mem)
    elapsed_time = time.time() - start_time

    return best_fitness, elapsed_time, total_memory_allocated

def run_comparison_stats(file_path,  p_gen, p_one_cluster, p_two_cluster, p_clustering, optimum = 0,num_runs = 50):
    """
    Run comparison statistics for the Traveling Salesman Problem (TSP) using the BSO algorithm.

    Args:
        file_path (str): The path to the TSP file.
        p_gen (float): The probability of generating a new individual.
        p_one_cluster (float): The probability of selecting an individual from one cluster.
        p_two_cluster (float): The probability of selecting an individual from two clusters.
        p_clustering (float): The probability of clustering the individuals.
        optimum (float, optional): The known optimum for the TSP. Defaults to 0.
        num_runs (int, optional): The number of runs to perform. Defaults to 50.

    Returns:
        pandas.DataFrame: A DataFrame containing the average distance, standard deviation of distance, average duration,
        standard deviation of duration, average allocated memory, and standard deviation of memory for the BSO algorithm.

    """
    distances = []
    durations = []
    memory = []
    for i in range(num_runs):
      tour_distance, elapsed_time, total_memory_allocated = run_tsp_with_stats(file_path, optimum, p_gen, p_one_cluster, p_two_cluster, p_clustering)

      print('Run: ', i)
      print('Run Time: ', elapsed_time)

      distances.append(tour_distance)
      durations.append(elapsed_time)
      memory.append(total_memory_allocated / (1024 * 1024))


    # Calculate averages
    avg_distance = np.average(distances)
    avg_duration = np.average(durations)
    avg_memory = np.average(memory)

    sd_distance = np.std(distances)
    sd_duration = np.std(durations)
    sd_memory = np.std(memory)


    # Create a DataFrame
    data = {
        'Algorithm': ['BSO'],
        'Average Distance': [avg_distance],
        "SD Distance": [sd_distance],
        'Average Duration (s)': [avg_duration],
        "SD Duration": [sd_duration],
        'Average Allocated Memory (MB)': [avg_memory],
        "SD Memory": [sd_memory]
    }

    df = pd.DataFrame(data)

    # Print the DataFrame
    return df