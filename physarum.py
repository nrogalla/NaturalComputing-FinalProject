import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.utils import compute_permutation_distance
import matplotlib.pyplot as plt
import warnings
import networkx as nx
import random
import pandas as pd
import tsplib95
import itertools
import time
import tracemalloc
from SlimeMould.slime.dish import Dish

warnings.filterwarnings("ignore")

def load_tsp(file_path, show_graph = False):
  """
    Reads a Traveling Salesman Problem (TSP) instance from a file, parses the data, and returns the city coordinates
    and adjacency matrix representing the distances between cities. Optionally displays the graph of the TSP problem.

    Parameters:
    -----------
    file_path : str
        The path to the file containing the TSP instance data.

    show_graph : bool, optional (default: False)
        If True, displays a graph representing the TSP instance.

    Returns:
    --------
    cities : pandas.DataFrame
        A DataFrame containing the coordinates of the cities with three columns:
        - 'x': The x-coordinate of the city.
        - 'y': The y-coordinate of the city.
        - 'value': A constant value (4) for all cities.

    full_matrix : numpy.matrix
        The adjacency matrix representing the distances between cities.
  """
  with open(file_path) as f:
   text = f.read()

  problem = tsplib95.parse(text)
  full_matrix = nx.adjacency_matrix(problem.get_graph()).todense()
  graph = np.array(full_matrix)
  if not problem.node_coords:
    cities = pd.DataFrame(problem.display_data).T
  else:
    cities = pd.DataFrame(problem.node_coords).T
  cities.rename(columns={0: 'x', 1: 'y'}, inplace=True)
  cities['value'] = 4
  cities = cities.astype(int)
  position_dict = {index-1: (row['x'], row['y']) for index, row in cities.iterrows()}

  if show_graph == True:
    G = nx.from_numpy_array(graph)
    fig = plt.figure(figsize=(6.3, 5))
    nx.draw_networkx_nodes(G, position_dict,
            node_color='C1',
            node_shape='s',
            node_size=12)
  return cities,full_matrix


def phase_1_physarum(cities, show_graph = False):
  """
    Simulates the first phase of a Physarum polycephalum-inspired algorithm on a set of cities, using a Dish model
    to represent the growth of the slime mold. The algorithm starts from a random city and grows the mold until a
    specified number of iterations without reaching new cities is met or all cities are reached.

    Parameters:
    -----------
    cities : pandas.DataFrame
        A DataFrame containing the coordinates of the cities with three columns:
        - 'x': The x-coordinate of the city.
        - 'y': The y-coordinate of the city.
        - 'value': A constant value (4) for all cities.

    show_graph : bool, optional (default: False)
        If True, displays a graph representing the growth of the Physarum polycephalum on the cities.

    Returns:
    --------
    graph_phase1 : networkx.Graph
        The graph representing the connections formed by the slime mold during the first phase.

    pos_phase_1 : dict
        A dictionary representing the positions of the nodes in the graph.
  """
  start_index = random.choice(cities.index)
  start_loc = (cities.at[start_index,'x'], cities.at[start_index,'y'])
  dish = Dish(dish_shape=(int(max(cities.x)) + 10, int(max(cities.y)) + 10), foods=cities, start_loc=start_loc, mould_shape=(5, 5), init_mould_coverage=1, decay=0.2)
  num_cities_reached = 0

  num_iter_without_add_city = 0
  max_num_iter_without_add_city = 20
  old_num_cities_reached = 0
  while num_cities_reached < len(cities):
    dish.mould.evolve()
    num_cities_reached = dish.mould.get_total_reached_foods()[-1]
    if num_cities_reached <= old_num_cities_reached:
      num_iter_without_add_city +=1

    else:
      num_iter_without_add_city = 0
    old_num_cities_reached = num_cities_reached
    if num_iter_without_add_city == max_num_iter_without_add_city:
      break

  graph_phase1, pos_phase_1 = dish.food_graph, dish.food_positions

  if show_graph == True:
    fig = plt.figure(figsize=(6.3, 5))
    nx.draw(graph_phase1, pos_phase_1,
            node_color='C1',
            node_shape='s',
            node_size=12,
            with_labels=True)
  return graph_phase1, pos_phase_1

def remove_edges_exceeding_degree_or_forming_cycles_longest(graph_original, distance_matrix):
    """
    Removes edges from the graph represented by an adjacency matrix to ensure that no node exceeds a degree of 2
    and no cycles are formed, preferring to remove edges with the largest distance.

    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        The adjacency matrix of the graph from which edges will be removed.

    Returns:
    --------
    np.ndarray
        The modified adjacency matrix with edges removed according to the specified rules.
    """
    # Create a graph from the adjacency matrix
    graph_copy = graph_original.copy()

    # Get the list of edges with their weights
    edges_with_weights = [(u, v, distance_matrix[u-1, v-1]) for u, v in graph_copy.edges]

    # Sort edges by weight in descending order
    edges_sorted_by_weight = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)

    for u, v, weight in edges_sorted_by_weight:
        if graph_copy.degree(u) > 2 or graph_copy.degree(v) > 2:

            graph_copy.remove_edge(u, v)
        else:
            cycles = nx.cycle_basis(graph_copy)
            for cycle in cycles:

              if u in cycle and v in cycle and graph_copy.has_edge(u,v):
                graph_copy.remove_edge(u, v)

                continue
    return graph_copy

def get_removed_graph_longest(graph_phase1,distance_matrix, pos_phase_1, show_graph = False):
  """
    Generates a graph by removing longest edges that exceed node degree of 2 or form cycles and optionally displays it.

    Parameters:
    -----------
    graph_phase1 : networkx.Graph
        The initial graph from which edges will be removed.

    pos_phase_1 : dict
        A dictionary representing the positions of the nodes in the graph.

    show_graph : bool, optional (default: False)
        If True, displays the resulting graph after edge removal.

    Returns:
    --------
    networkx.Graph
        The graph after removing edges according to the specified rules.
  """
  G_rem = remove_edges_exceeding_degree_or_forming_cycles_longest(graph_phase1, distance_matrix)
  
  if show_graph == True:
    fig = plt.figure(figsize=(6.3, 5))
    nx.draw(G_rem,pos_phase_1,
            node_color='C1',
            node_shape='s',
            node_size=12,
            with_labels=True)
  return G_rem


def tsp_full_tour(graph):
    """
    Generates a full tour for the Traveling Salesman Problem (TSP) by adding edges to the graph while ensuring that
    each node has a degree of 2 and no cycles are formed.

    Parameters:
    -----------
    graph : networkx.Graph
        The input graph representing the cities and their connections.

    Returns:
    --------
    networkx.Graph
        A copy of the input graph with additional edges to form a full tour for the TSP.
    """
    # Create a copy of the graph to modify
    full_graph = graph.copy()

    nodes = list(graph.nodes())
    node_pairs = list(itertools.permutations(nodes, 2))

    for pair in node_pairs:
        node1, node2 = pair
        if node1 == node2:
          continue

        # Check if an edge already exists between the nodes
        if not graph.has_edge(node1, node2):
            test_graph = full_graph.copy()
            test_graph.add_edge(node1, node2)

            if all(degree <= 2 for node, degree in test_graph.degree()) and not nx.cycle_basis(test_graph):

                full_graph.add_edge(node1, node2)

    for pair in node_pairs:
      node1, node2 = pair
      if full_graph.degree(node1) == 1 and full_graph.degree(node2) == 1:
         full_graph.add_edge(node1, node2)

    return full_graph

def get_physarum_tsp_solution(G_rem, pos_phase_1, show_graph = False):
    """
    Computes the solution to the Traveling Salesman Problem (TSP) using the Physarum polycephalum-inspired algorithm
    on the given graph and optionally displays the solution.

    Parameters:
    -----------
    G_rem : networkx.Graph
        The graph representing the cities and their connections after removing excess edges.

    pos_phase_1 : dict
        A dictionary representing the positions of the nodes in the graph.

    show_graph : bool, optional (default: False)
        If True, displays the TSP solution graph.

    Returns:
    --------
    list
        The sequence of nodes representing the TSP solution.

    """
    graph_phase1 = G_rem.copy()

    full_tour_graph  = tsp_full_tour(graph_phase1)
    if show_graph == True:
      fig = plt.figure(figsize=(6.3, 5))
      nx.draw(full_tour_graph,pos_phase_1,
              node_color='C1',
              node_shape='s',
              node_size=12,
              with_labels=True)
    return nx.cycle_basis(full_tour_graph)[0]


def tsp_full_tour_nearest(graph, distance_matrix):
    """
    Generates a full tour for the Traveling Salesman Problem (TSP) by adding edges to the graph while ensuring that
    each node has a degree of 2 and no cycles are formed.

    Parameters:
    -----------
    graph : networkx.Graph
        The input graph representing the cities and their connections.

    Returns:
    --------
    networkx.Graph
        A copy of the input graph with additional edges to form a full tour for the TSP.
    """
    # Create a copy of the graph to modify
    full_graph = graph.copy()

    nodes = list(graph.nodes())
    node_pairs = list(itertools.permutations(nodes, 2))
    edges_with_weights = [(u, v, distance_matrix[u-1, v-1]) for u, v in node_pairs]
    edges_sorted_by_weight = sorted(edges_with_weights, key=lambda x: x[2])

    for node1, node2, weight in edges_sorted_by_weight:

        # Check if an edge already exists between the nodes
        if not graph.has_edge(node1, node2):
            test_graph = full_graph.copy()
            test_graph.add_edge(node1, node2)


            if all(degree <= 2 for node, degree in test_graph.degree()) and not nx.cycle_basis(test_graph):

                full_graph.add_edge(node1, node2)

    for node1, node2, weight in edges_sorted_by_weight:

      if full_graph.degree(node1) == 1 and full_graph.degree(node2) == 1:
         full_graph.add_edge(node1, node2)

    return full_graph

def get_physarum_tsp_solution_nearest(G_rem, pos_phase_1, distance_matrix, show_graph = False):
    """
    Computes the solution to the Traveling Salesman Problem (TSP) using the Physarum polycephalum-inspired algorithm
    on the given graph and optionally displays the solution.

    Parameters:
    -----------
    G_rem : networkx.Graph
        The graph representing the cities and their connections after removing excess edges.

    pos_phase_1 : dict
        A dictionary representing the positions of the nodes in the graph.

    show_graph : bool, optional (default: False)
        If True, displays the TSP solution graph.

    Returns:
    --------
    list
        The sequence of nodes representing the TSP solution.

    """
    graph_phase1 = G_rem.copy()

    full_tour_graph  = tsp_full_tour_nearest(graph_phase1, distance_matrix)
    if show_graph == True:
      fig = plt.figure(figsize=(6.3, 5))
      nx.draw(full_tour_graph,pos_phase_1,
              node_color='C1',
              node_shape='s',
              node_size=12,
              with_labels=True)

    return nx.cycle_basis(full_tour_graph)[0]


def get_optimized_physarum_solution(distance_matrix, physarum_tour, log_file = None):

    initial_permutation = [node - 1 for node in physarum_tour]
    # Solve the TSP using the simulated annealing algorithm
    permutation_phys_siman, distance_phys_siman = solve_tsp_simulated_annealing(
        distance_matrix, x0=initial_permutation, log_file = log_file
    )
    return [city+1 for city in permutation_phys_siman], distance_phys_siman

# Solve the TSP using the simulated annealing algorithm
def get_sim_an_solution(distance_matrix, log_file = None):
  permutation_siman, distance_siman = solve_tsp_simulated_annealing(
      distance_matrix, log_file = log_file
  )
  return [city+1 for city in permutation_siman], distance_siman


def get_graph_from_path(tsp_solution):
  G = nx.Graph()

  # Add nodes and edges for the TSP solution
  edges = [(tsp_solution[i], tsp_solution[i + 1]) for i in range(len(tsp_solution) - 1)]
  edges.append((tsp_solution[0],tsp_solution[-1] ))
  G.add_edges_from(edges)
  return G



def plot_comparison(full_tour_graph, permutation_phys_siman, permutation_siman, pos_phase_1, tour_distance, distance_phys_siman, distance_siman):

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  nx.draw(get_graph_from_path(full_tour_graph),pos_phase_1,
          node_color='C1',
          node_shape='s',
          node_size=12,
          with_labels=True,
          ax=axes[0])

  axes[0].set_title('Initial Physarum Solution')
  nx.draw(get_graph_from_path(permutation_phys_siman),pos_phase_1,
          node_color='C1',
          node_shape='s',
          node_size=12,
          with_labels=True,
          ax=axes[1])

  axes[1].set_title('Simulated Annealing with Physarum Initialisation')
  nx.draw(get_graph_from_path(permutation_siman),pos_phase_1,
          node_color='C1',
          node_shape='s',
          node_size=12,
          with_labels=True,
          ax=axes[2])

  axes[2].set_title('Simulated Annealing Solution')

  # Add distance text below each subplot
  fig.text(0.18, 0.05, f'Distance: {tour_distance}', ha='center')
  fig.text(0.50, 0.05, f'Distance: {distance_phys_siman}', ha='center')
  fig.text(0.83, 0.05, f'Distance: {distance_siman}', ha='center')

  plt.tight_layout(rect=[0, 0.1, 1, 1])  

  plt.show()


def run_tsp(file_path, physarum_method = "nearest", show_graph = True):
    """
    Runs the Traveling Salesman Problem (TSP) solution process using the Physarum polycephalum-inspired algorithm
    and Simulated Annealing, and optionally displays a comparison plot.

    Parameters:
    -----------
    file_path : str
        The path to the file containing the TSP instance data.

    physarum_method : str, optional (default: "nearest")
        The method to use for solving the TSP using the Physarum polycephalum-inspired algorithm.
        Available options: "nearest" for nearest neighbor heuristic, "simple" for the original method.

    show_graph : bool, optional (default: True)
        If True, displays a comparison plot of the TSP solutions.

    Returns:
    --------
    tuple or None
        If show_graph is False, returns a tuple containing the tour distance for the initial Physarum solution,
        the optimized Physarum solution distance, and the Simulated Annealing solution distance.
        If show_graph is True, returns None.
    """
    #Load
    cities, distance_matrix = load_tsp(file_path)
    #Phase 1
    graph_phase1, pos_phase_1 = phase_1_physarum(cities)

    #Remove violating edges
    G_rem = get_removed_graph_longest(graph_phase1,distance_matrix, pos_phase_1)

    # Initial Physarum solution
    if physarum_method == "nearest":
      full_tour_graph = get_physarum_tsp_solution_nearest(G_rem, pos_phase_1, np.array(distance_matrix))
      #full_tour_graph = get_physarum_tsp_solution_nearest(G_rem, pos_phase_1, np.array(distance_matrix))
    elif physarum_method == "simple" :
      full_tour_graph = get_physarum_tsp_solution(G_rem, pos_phase_1)
    tour_distance = compute_permutation_distance(distance_matrix, [city-1 for city in full_tour_graph])
    # Optimized Physarum Solution
    permutation_phys_siman, distance_phys_siman = get_optimized_physarum_solution(distance_matrix, full_tour_graph)
    # Simulated Annealing
    permutation_siman, distance_siman = get_sim_an_solution(distance_matrix)

    # plot
    if show_graph == True:
      plot_comparison(full_tour_graph, permutation_phys_siman, permutation_siman, pos_phase_1, tour_distance, distance_phys_siman, distance_siman)
    else:
      return tour_distance, distance_phys_siman, distance_siman
    
def run_tsp_with_stats(file_path, physarum_method="nearest", show_graph=True):
    """
    Runs the Traveling Salesman Problem (TSP) solution process using the Physarum polycephalum-inspired algorithm,
    Simulated Annealing, and computes memory allocation and execution time statistics.

    Parameters:
    -----------
    file_path : str
        The path to the file containing the TSP instance data.

    physarum_method : str, optional (default: "nearest")
        The method to use for solving the TSP using the Physarum polycephalum-inspired algorithm.
        Available options: "nearest" for nearest neighbor heuristic, "simple" for the original method.

    show_graph : bool, optional (default: True)
        If True, displays a comparison plot of the TSP solutions.

    Returns:
    --------
    tuple or None
        If show_graph is False, returns a tuple containing the tour distance for the initial Physarum solution,
        the optimized Physarum solution distance, the Simulated Annealing solution distance, and statistics
        including execution time and memory allocation.
        If show_graph is True, returns None.
    """
    # Load TSP data
    cities, distance_matrix = load_tsp(file_path)

    # Initial Physarum Solution
    start_time = time.time()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Phase 1
    graph_phase1, pos_phase_1 = phase_1_physarum(cities)
    # Phase 2
    #G_rem = get_removed_graph(graph_phase1, pos_phase_1)
    G_rem = get_removed_graph_longest(graph_phase1,distance_matrix, pos_phase_1)

    if physarum_method == "nearest":
        full_tour_graph = get_physarum_tsp_solution_nearest(G_rem, pos_phase_1, np.array(distance_matrix))
    elif physarum_method == "simple":
        full_tour_graph = get_physarum_tsp_solution(G_rem, pos_phase_1)
    tour_distance = compute_permutation_distance(distance_matrix, [city-1 for city in full_tour_graph])

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats_initial = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory_allocated_initial = sum(stat.size_diff for stat in stats_initial)
    elapsed_time_initial_physarum = time.time() - start_time

    # Optimized Physarum Solution
    start_time = time.time()
    tracemalloc.start()
    snapshot3 = tracemalloc.take_snapshot()

    permutation_phys_siman, distance_phys_siman = get_optimized_physarum_solution(distance_matrix, full_tour_graph)

    snapshot4 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats_optimized = snapshot4.compare_to(snapshot3, 'lineno')
    total_memory_allocated_optimized = total_memory_allocated_initial +sum(stat.size_diff for stat in stats_optimized)
    elapsed_time_optimized_physarum = elapsed_time_initial_physarum + time.time() - start_time

    # Simulated Annealing
    start_time = time.time()
    tracemalloc.start()
    snapshot5 = tracemalloc.take_snapshot()

    permutation_siman, distance_siman = get_sim_an_solution(distance_matrix)

    snapshot6 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats_sim_ann = snapshot6.compare_to(snapshot5, 'lineno')
    total_memory_allocated_sim_ann = sum(stat.size_diff for stat in stats_sim_ann)
    elapsed_time_sim_ann = time.time() - start_time


    all_stats = elapsed_time_initial_physarum, elapsed_time_optimized_physarum, elapsed_time_sim_ann, total_memory_allocated_initial  / (1024 ** 2), total_memory_allocated_optimized  / (1024 ** 2), total_memory_allocated_sim_ann  / (1024 ** 2)

    # Plot or return results
    if show_graph:
        plot_comparison(full_tour_graph, permutation_phys_siman, permutation_siman, pos_phase_1, tour_distance, distance_phys_siman, distance_siman)
    else:
        return tour_distance, distance_phys_siman, distance_siman, all_stats



def plot_log_graph():
  def extract_values_from_log(file_path):
      current_values = []

      with open(file_path, 'r') as file:
          for line in file:
              if "Current value" in line:
                  # Split the line by space and extract the current value
                  parts = line.split()

                  current_value = float(parts[4])
                  current_values.append(current_value)

      return current_values

  # Path to the log file
  log_file_path = "/content/log_opt.txt"

  # Extract current values from the log file
  values_opt = extract_values_from_log(log_file_path)
  # Path to the log file
  log_file_path = "/content/log_simann.txt"

  # Extract current values from the log file
  values_sim_ann = extract_values_from_log(log_file_path)
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))

  # Plot the current values by iteration
  axes[0].plot(values_opt)
  axes[0].set_xlabel('Iteration')
  axes[0].set_ylabel('Distance')
  axes[0].set_title('Distance by Iteration for Optimized Physarum Algorithm')

  axes[1].plot(values_sim_ann)
  axes[1].set_xlabel('Iteration')
  axes[1].set_ylabel('Distance')
  axes[1].set_title('Distance by Iteration for Simulated Annealing Algorithm')

  plt.show()



def run_tsp_log_graph(file_path, physarum_method = "nearest", show_graph = True):
  """
    Runs the Traveling Salesman Problem (TSP) solution process using the Physarum polycephalum-inspired algorithm,
    Simulated Annealing, and optionally displays a comparison plot, along with logs.

    Parameters:
    -----------
    file_path : str
        The path to the file containing the TSP instance data.

    physarum_method : str, optional (default: "nearest")
        The method to use for solving the TSP using the Physarum polycephalum-inspired algorithm.
        Available options: "nearest" for nearest neighbor heuristic, "simple" for the original method.

    show_graph : bool, optional (default: True)
        If True, displays a comparison plot of the TSP solutions and log graphs.

    Returns:
    --------
    tuple or None
        If show_graph is False, returns a tuple containing the tour distance for the initial Physarum solution,
        the optimized Physarum solution distance, and the Simulated Annealing solution distance.
        If show_graph is True, returns None.
  """

  #Load
  cities, distance_matrix = load_tsp(file_path)
  #Phase 1
  graph_phase1, pos_phase_1 = phase_1_physarum(cities)
  #Remove violating edges

  G_rem = get_removed_graph_longest(graph_phase1,distance_matrix, pos_phase_1)
  # Initial Physarum solution

  full_tour_graph = None
  if physarum_method == "nearest":
    full_tour_graph = get_physarum_tsp_solution_nearest(G_rem, pos_phase_1, np.array(distance_matrix))
  elif physarum_method == "simple" :
    full_tour_graph = get_physarum_tsp_solution(G_rem, pos_phase_1)
  tour_distance = compute_permutation_distance(distance_matrix, [city-1 for city in full_tour_graph])
  # Optimized Physarum Solution
  permutation_phys_siman, distance_phys_siman = get_optimized_physarum_solution(distance_matrix, full_tour_graph, "/content/log_opt.txt")
  # Simulated Annealing
  permutation_siman, distance_siman = get_sim_an_solution(distance_matrix, "/content/log_simann.txt")

  # plot
  if show_graph == True:
    plot_comparison(full_tour_graph, permutation_phys_siman, permutation_siman, pos_phase_1, tour_distance, distance_phys_siman, distance_siman)
    plot_log_graph()
  else:
    return tour_distance, distance_phys_siman, distance_siman


def run_comparison_stats(file_path, physarum_method = "nearest",num_runs = 50):
    """
    Runs multiple iterations of the Traveling Salesman Problem (TSP) solution process using the Physarum polycephalum-inspired
    algorithm and Simulated Annealing to collect statistics on distance, execution time, and memory allocation.

    Parameters:
    -----------
    file_path : str
        The path to the file containing the TSP instance data.

    physarum_method : str, optional (default: "nearest")
        The method to use for solving the TSP using the Physarum polycephalum-inspired algorithm.
        Available options: "nearest" for nearest neighbor heuristic, "simple" for the original method.

    num_runs : int, optional (default: 50)
        The number of times to run each algorithm to collect statistics.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the average distance, duration, and allocated memory for each algorithm over multiple runs.
    """
    initial_physarum_distances = []
    optimized_physarum_distances = []
    sim_ann_distances = []
    durations_initial = []
    durations_otpimized = []
    durations_simann = []
    memory_initial = []
    memory_optimized = []
    memory_simann = []
    for _ in range(num_runs):

      tour_distance, distance_phys_siman, distance_siman, all_stats = run_tsp_with_stats(file_path, physarum_method, False)
      initial_physarum_distances.append(tour_distance)
      optimized_physarum_distances.append(distance_phys_siman)
      sim_ann_distances.append(distance_siman)

      # stats
      elapsed_time_initial_physarum, elapsed_time_optimized_physarum, elapsed_time_sim_ann, total_memory_allocated_initial, total_memory_allocated_optimized, total_memory_allocated_sim_ann = all_stats

      durations_initial.append(elapsed_time_initial_physarum)
      durations_otpimized.append(elapsed_time_optimized_physarum)
      durations_simann.append(elapsed_time_sim_ann)
      memory_initial.append(total_memory_allocated_initial)
      memory_optimized.append(total_memory_allocated_optimized)
      memory_simann.append(total_memory_allocated_sim_ann)


    # Calculate averages
    avg_initial_distance = np.average(initial_physarum_distances)
    avg_initial_duration = np.average(durations_initial)
    avg_initial_memory = np.average(memory_initial)

    avg_optimized_distance = np.average(optimized_physarum_distances)
    avg_optimized_duration = np.average(durations_otpimized)
    avg_optimized_memory = np.average(memory_optimized)

    avg_simann_distance = np.average(sim_ann_distances)
    avg_simann_duration = np.average(durations_simann)
    avg_simann_memory = np.average(memory_simann)


    data = {
        'Algorithm': ['Initial Physarum', 'Optimized Physarum', 'Sim Ann'],
        'Average Distance': [avg_initial_distance, avg_optimized_distance, avg_simann_distance],
        'Average Duration (s)': [avg_initial_duration, avg_optimized_duration, avg_simann_duration],
        'Average Allocated Memory (MB)': [avg_initial_memory, avg_optimized_memory, avg_simann_memory]
    }

    df = pd.DataFrame(data)

    return df