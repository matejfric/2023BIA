from typing import Any
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import copy
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics.simulated_annealing import solve_tsp_simulated_annealing
from tqdm import tqdm


class TSP:
    """
    Traveling Salesman Problem (TSP) solver using various algorithms.

    This class provides functionality to solve the TSP for a given set of cities.
    It supports both exact and heuristic algorithms for solving the problem.

    Args:
        n_cities (int, optional): The number of cities to include in the problem instance.
            Defaults to 5.

    Attributes:
        cities (numpy.ndarray): An array of city coordinates.
        _dist_mat (numpy.ndarray): An internal variable to store the distance matrix.

    Methods:
        dist_mat (property): Get the distance matrix for the cities.
        show(): Display a scatter plot of the cities on a 2D plane.
        solve(): Solve the TSP using either dynamic programming or simulated annealing.

    Static Methods:
        ulysses(n_cities: int): Load city coordinates from the Ulysses22 dataset and compute the distance matrix.

    Note:
        - For small instances (n_cities < 15), the solve() method uses dynamic programming.
        - For 22 cities, it uses an optimal solution.
        - For other cases, it employs simulated annealing.
    """

    def __init__(self, n_cities: int = 5) -> None:
        """
        Initialize a TSP instance with a given number of cities.

        Args:
            n_cities (int, optional): The number of cities to include in the problem instance.
                Defaults to 5.
        """
        distance_matrix, cities = TSP.ulysses(n_cities)
        self.cities = cities
        # self.cities = np.array([np.random.uniform(0, 1000, 2) for _ in range(n_cities)], dtype=np.float64)
        self._dist_mat = distance_matrix

    @property
    def dist_mat(self) -> np.ndarray:
        if self._dist_mat is None:
            self._dist_mat = distance_matrix(self.cities, self.cities)
        return self._dist_mat

    @dist_mat.setter
    def dist_mat(self, value: Any):
        raise AttributeError("'dist_mat' is a read-only property")

    def show(self) -> None:
        """
        Display a scatter plot of the cities on a 2D plane.
        """
        plt.figure()
        cities = self.cities
        plt.scatter(cities[:, 0], cities[:, 1])
        plt.show()

    def solve(self) -> tuple:
        """
        Solve the TSP using dynamic programming (for small instances) or simulated annealing (for larger instances).

        Returns:
            tuple: A tuple containing the total distance of the tour and the order of visited cities.
        """
        if len(self.cities) < 15:
            print("Optimal solution:")
            permutation, distance = solve_tsp_dynamic_programming(
                self.dist_mat)
        elif len(self.cities) == 22:
            ullyses22_opt = [0, 13, 12, 11, 6, 5, 14, 4, 10,
                             8, 9, 18, 19, 20, 15, 2, 1, 16, 21, 3, 17, 7]
            permutation, distance = ullyses22_opt, -1
        else:
            print("Simulated Annealing solution:")
            permutation, distance = solve_tsp_simulated_annealing(
                self.dist_mat)

        print(distance)
        print(permutation)

        return distance, permutation

    @staticmethod
    def ulysses(n_cities: int = 4) -> tuple:
        """
        Load city coordinates from the Ulysses22 dataset and compute the distance matrix.

        Args:
            n_cities (int, optional): The number of cities to load from the dataset. Defaults to 4.

        Returns:
            tuple: A tuple containing the distance matrix and an array of city coordinates.
        """
        coordinates = {}
        n = 0
        with open('ulysses22.tsp.txt', 'r') as file:
            in_node_coords = False
            for line in file:
                if line.strip() == "NODE_COORD_SECTION":
                    in_node_coords = True
                    continue
                if in_node_coords:
                    if n >= n_cities or line.strip() == "EOF":
                        break
                    parts = line.strip().split()
                    node = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates[node] = (x, y)
                    n += 1

        # Calculate the distance matrix
        dimension = len(coordinates)
        distance_matrix = np.zeros((dimension, dimension), dtype=np.float64)

        for i in range(1, dimension + 1):
            for j in range(1, dimension + 1):
                distance_matrix[i - 1, j -
                                1] = euclidean(coordinates[i], coordinates[j])

        # Print the distance matrix
        # for row in distance_matrix:
        #     print(" ".join(map(str, row)))

        cities = np.array(list(coordinates.values()))

        return distance_matrix, cities


class GeneticAlgorithm:
    """
    Genetic Algorithm (GA) for solving the Traveling Salesman Problem (TSP).

    This class provides methods for solving the TSP using a genetic algorithm approach.
    It includes functions for generating cities and distances, creating an initial population,
    performing crossover, mutation, and evaluating fitness. It also offers a visualization of
    the TSP and the final solution.

    Methods:
        generate_cities(self, dimension: int) -> None:
        generate_distances(self, dimension: int) -> None:
        generate_initial_population(self, dimension: int, n_individuals: int) -> list:
        crossover(self, parent1: list, parent2: list) -> list:
        mutate(self, offspring: list, mutation_rate: float) -> list:
        calc_distance(self, individual: list) -> float:
        eval_cost(self, population: list) -> list:
        cost2fitness(self, costs: list) -> list:
        rank_fitness(self, fitness: list) -> list:
        select_parent(self, parent: list, population: list, fitness: list) -> list:
        plot_optimal_route(self, ax: plt.Axes, permutation: list) -> None:
        plot_individual(self, ax: plt.Axes, individual: list, permutation: list) -> None:
        plot_cities(self, ax: plt.Axes) -> None:
        run(self, n_individuals: int = 10, n_generations: int = 200, dimension: int = 5,
            max_mutation_rate: float = 0.99, min_mutation_rate: float = 0.05, alpha: float = 0.95) -> None:
    """

    def generate_cities(self, dimension: int) -> np.ndarray:
        """
        Generate city coordinates for the TSP instance.

        Args:
            dimension (int): The number of cities in the TSP.
        """
        tsp = TSP(dimension)
        self.cities = tsp.cities

    def generate_distances(self, dimension: int) -> np.ndarray:
        """
        Generate the distance matrix for the TSP based on city coordinates.

        Args:
            dimension (int): The number of cities in the TSP.
        """
        tsp = TSP(dimension)
        self.dist_mat = tsp.dist_mat

    def generate_initial_population(self, dimension: int, n_individuals: int) -> list:
        """
        Generate an initial population of TSP solutions.

        Args:
            dimension (int): The number of cities in the TSP.
            n_individuals (int): Number of individuals in the population.

        Returns:
            list: A list of TSP solutions (individuals).
        """
        population = []
        for _ in range(n_individuals):
            array = np.arange(dimension)
            np.random.shuffle(array)
            population.append(array)
        return population

    def crossover(self, parent1: list, parent2: list) -> list:
        """
        Perform crossover (recombination) to create a new individual from two parents.

        Args:
            parent1 (list): First parent individual.
            parent2 (list): Second parent individual.

        Returns:
            list: The offspring individual resulting from crossover.

        Notes:
            - This is a naive implementation. 
            - There are better known solutions, e.g., 
                - Cyclic Crossover (CX),
                - Partial-Mapped Crossover (PMX),
                - Order Crossover (OX),
        """
        n_cities = len(parent1)
        start = int(np.random.uniform(n_cities))
        end = int(np.random.uniform(start+1, n_cities))
        offspring = list(parent1[start:end])
        for city in parent2:
            if city not in offspring:
                offspring.append(city)
        return offspring

    def _swap(self, list_: list, idx1: int, idx2: int):
        list_[idx1], list_[idx2] = list_[idx2], list_[idx1]
        return list_

    def mutate(self, offspring: list, mutation_rate: float) -> list:
        """
        Apply mutation to an individual with a given mutation rate.

        Args:
            offspring (list): The individual to mutate.
            mutation_rate (float): The probability of mutation.

        Returns:
            list: The mutated individual.
        """
        n_cities = len(offspring)
        for _ in offspring:
            # Swap two random neighbors
            if np.random.uniform() < mutation_rate:
                idx1 = int(np.random.uniform(n_cities))
                idx2 = (idx1+1) % n_cities
                offspring = self._swap(offspring, idx1, idx2)
        return offspring

    def calc_distance(self, individual: list) -> float:
        """
        Calculate the total distance of a TSP solution (individual).

        Args:
            individual (list): A TSP solution represented as a permutation of city indices.

        Returns:
            float: The total distance of the TSP tour for the individual.
        """
        total_dst = 0.0
        dist_mat = self.dist_mat
        for idx in range(len(individual)-1):
            i = individual[idx]
            j = individual[idx + 1]
            total_dst += dist_mat[i, j]

        # From the last city back home
        total_dst += dist_mat[individual[-1], individual[0]]

        return total_dst

    def eval_cost(self, population: list) -> list:
        """
        Evaluate the cost (distance) of each individual in the population.

        Args:
            population (list): The population of TSP solutions (individuals).

        Returns:
            list: A list of distances, one for each individual in the population.
        """
        costs = []
        for individual in population:
            costs.append(
                self.calc_distance(individual)
            )
        return costs

    def cost2fitness(self, costs: list) -> list:
        """
        Convert cost values to fitness values for a population of individuals.

        Args:
            costs (list): A list of cost values (e.g., distances).

        Returns:
            list: A list of fitness values corresponding to the cost values.
        """

        # Over-complicated formula from the lecture:
        """
        eps = 0.01
        fitness = []
        f_min = np.min(costs)
        f_max = np.max(costs)
        denom = f_min - f_max
        nomin = f_min*eps-f_max
        one_minus_eps = 1-eps
        for cost in costs:
            fitness.append(
                (one_minus_eps * cost + nomin) / denom
            )
        """

        # Simple conversion
        eps = 0.01
        fitness = []
        for cost in costs:
            fitness.append(
                1 / (cost + eps)
            )

        # Exponential fitness
        """
        fitness = []
        for cost in costs:
            fitness.append(
                1 / (np.power(cost, 2) + 1)
            )
        """
        return fitness

    def rank_fitness(self, fitness: list) -> list:
        """
        Rank the fitness values of individuals in the population,
        i.e., prepare fitness values for Rank Selection.

        Args:
            fitness (list): A list of fitness values for the population.

        Returns:
            list: A list of ranked fitness values normalized to [0, 1].
        """
        # Add indexes and sort value (fitness)
        indexed_fitness = list(enumerate(fitness))
        indexed_fitness.sort(key=lambda x: x[1])

        # Create a map between index and new value (rank)
        fitneess_map = {}
        for i, (idx, _) in enumerate(indexed_fitness):
            fitneess_map[idx] = i

        transformed_fitness = [fitneess_map[i]
                               for i in range(len(fitness))]

        # Normalize to [0,1]
        fitness_total = 0.0
        for val in transformed_fitness:
            fitness_total += val
        ranks = [val / fitness_total for val in transformed_fitness]

        return ranks

    def _pick_one(self, population: list, ranks: list) -> list:
        """
        Select an individual from the population based on its rank.

        Args:
            population (list): The population of TSP solutions.
            ranks (list): A list of ranked fitness values corresponding to individuals.

        Returns:
            list: The selected individual based on its rank.
        """
        idx = 0
        r = np.random.uniform()
        while r > 0:
            r -= ranks[idx]
            idx += 1
        return population[idx - 1]

    def select_parent(self, parent: list, population: list, fitness: list) -> list:
        """
        Select a parent individual for reproduction based on fitness values.

        Args:
            parent (list): The current parent individual.
            population (list): The population of TSP solutions.
            fitness (list): A list of fitness values for the population.

        Returns:
            list: The selected parent for reproduction.
        """
        ranks = self.rank_fitness(fitness)
        new_parent = self._pick_one(population, ranks)

        # Make sure that the new parent is different from the other
        eq = True
        while eq:
            for idx in range(len(parent)):
                if parent[idx] != new_parent[idx]:
                    eq = False
                    break
            if eq:
                new_parent = self._pick_one(population, ranks)

        return new_parent

    def plot_optimal_route(self, ax: plt.Axes, permutation: list) -> None:
        """
        Plot the optimal TSP route on a given axis.

        Args:
            ax (plt.Axes): The axis on which to visualize the TSP route.
            permutation (list): The optimal TSP tour represented as a permutation of city indices.
        """
        cities = self.cities
        # Plot lines to connect cities in the TSP order
        for i in range(len(permutation) - 1):
            start_city = cities[permutation[i]]
            end_city = cities[permutation[i + 1]]
            ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
                    'g', alpha=0.5, linewidth=4)

        # Connect the last city to the first city to complete the TSP loop
        start_city = cities[permutation[-1]]
        end_city = cities[permutation[0]]
        ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
                'g', alpha=0.5, linewidth=4, label='Optimal Route')

    def plot_individual(self, ax: plt.Axes, individual: list, permutation: list) -> None:
        """
        Plot an individual's TSP route on a given axis.

        Args:
            ax (plt.Axes): The axis on which to visualize the TSP route.
            individual (list): The TSP solution to be visualized.
            permutation (list): The optimal TSP tour represented as a permutation of city indices.
        """
        cities = self.cities

        # Clear the current axis to remove previous routes
        ax.cla()
        self.plot_cities(ax)
        self.plot_optimal_route(ax, permutation)

        # Plot lines to connect cities in the TSP order
        for i in range(len(individual) - 1):
            start_city = cities[individual[i]]
            end_city = cities[individual[i + 1]]
            ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
                    'r', alpha=0.8, linewidth=3)

        # Connect the last city to the first city to complete the TSP loop
        start_city = cities[individual[-1]]
        end_city = cities[individual[0]]
        ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
                'r', alpha=0.8, linewidth=3, label='Fittest Individual')

        plt.title('TSP Visualization')
        plt.legend()
        # Pause to display the new plot
        plt.pause(0.01)

    def plot_cities(self, ax: plt.Axes) -> None:
        """
        Plot city coordinates on a given axis.

        Args:
            ax (plt.Axes): The axis on which to visualize city coordinates.
        """
        cities = self.cities
        # Scatter plot the cities (with labels)
        ax.scatter(cities[:, 0], cities[:, 1], c='b', label='Cities')
        for i, city in enumerate(cities):
            ax.text(city[0], city[1], str(i),
                    fontsize=12, ha='center', va='bottom')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    def run(self, n_individuals: int = 10, n_generations: int = 200, dimension: int = 5,
            max_mutation_rate: float = 0.99, min_mutation_rate: float = 0.05, alpha: float = 0.95) -> None:
        """
        Run the Genetic Algorithm to solve the TSP and visualize the results.

        Args:
            n_individuals (int): Number of individuals in the population.
            n_generations (int): Number of generations for the GA.
            dimension (int): The number of cities in the TSP.
            max_mutation_rate (float): Maximum mutation rate.
            min_mutation_rate (float): Minimum mutation rate.
            alpha (float): Rate of mutation rate reduction.
        """

        # Load cities
        self.generate_cities(dimension)
        # Precompute distance matrix
        self.generate_distances(dimension)
        # Prepare plot
        _, ax = plt.subplots(1, 1)
        # Get optimal solution for comparison
        _, permutation_opt = TSP(dimension).solve()
        # Set mutation rate
        mutation_rate = max_mutation_rate
        # Generate initial population
        population = self.generate_initial_population(dimension, n_individuals)
        # Evaluate population fitness
        costs = self.eval_cost(population)
        fitness = self.cost2fitness(costs)
        def f(x): return self.calc_distance(x)

        # Repeat for number of generations
        for _ in tqdm(range(n_generations), "Genetic Algorithm"):
            # Offspring is always put into a new population
            new_population = copy.deepcopy(population)
            for individual in range(n_individuals):
                # Selection of parents
                parent1 = population[individual]
                parent2 = self.select_parent(parent1,
                                             population,
                                             fitness)
                # Crossover
                offspring = self.crossover(parent1, parent2)
                # Mutate
                offspring = self.mutate(offspring, mutation_rate)
                # Elitism
                if f(offspring) < f(parent1):
                    new_population[individual] = offspring
            # Simulated Annealing
            if mutation_rate > min_mutation_rate:
                mutation_rate *= alpha
            # Update population and evaluate fitness
            population = new_population
            costs = self.eval_cost(population)
            fitness = self.cost2fitness(costs)
            # Visualize
            self.plot_individual(
                ax, population[np.argmin(costs)], permutation_opt)

        # Extract solution
        solution_idx = np.argmin(costs)
        permutation = population[solution_idx]
        distance = costs[solution_idx]

        plt.savefig('my_tsp.png', dpi=300)

        print(f"GA:\n{distance}\n{permutation}")


class AntColonyOptimization:
    def __init__(self) -> None:
        pass


if __name__ == '__main__':
    np.random.seed(42)
    ga = GeneticAlgorithm()
    ga.run(dimension=12, n_individuals=10, n_generations=125)
    #ga.run(dimension=22, n_individuals=10, n_generations=500)

