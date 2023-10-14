import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import copy
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics.simulated_annealing import solve_tsp_simulated_annealing
from tqdm import tqdm


class TSP:
    def __init__(self, n_cities: int = 5) -> None:
        distance_matrix, cities = TSP.ulysses(n_cities)
        self.cities = cities
        # self.cities = np.array([np.random.uniform(0, 1000, 2) for _ in range(n_cities)], dtype=np.float64)
        self._dist_mat = distance_matrix
        self.adj_mat = None

    @property
    def dist_mat(self):
        if self._dist_mat is None:
            self._dist_mat = distance_matrix(self.cities, self.cities)
        return self._dist_mat

    @dist_mat.setter
    def dist_mat(self, value):
        raise AttributeError("'dist_mat' is a read-only property")

    def show(self):
        plt.figure()
        cities = self.cities
        plt.scatter(cities[:, 0], cities[:, 1])
        plt.show()

    def solve(self):
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
    def ulysses(n_cities: int = 4):
        # Read the TSP file and extract coordinates
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
    def __init__(self) -> None:
        """
        self.n_individuals = n_individuals  # Number of individuals
        self.n_generations = n_generations  # Number of generations
        self.dimension = dimension   # In TSP, D will be a number of cities
        """

    def generate_cities(self, dimension) -> np.ndarray:
        tsp = TSP(dimension)
        self.cities = tsp.cities

    def generate_distances(self, dimension) -> np.ndarray:
        tsp = TSP(dimension)
        self.dist_mat = tsp.dist_mat

    def generate_initial_population(self, dimension, n_individuals):
        population = []
        for _ in range(n_individuals):
            array = np.arange(dimension)
            np.random.shuffle(array)
            population.append(array)
        return population

    def crossover(self, parent1, parent2):
        """
        This is a naive implementation. 
        There are better known solutions, e.g., 
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
        # list_ = list(list_)
        list_[idx1], list_[idx2] = list_[idx2], list_[idx1]
        return list_

    def mutate(self, offspring, mutation_rate):
        n_cities = len(offspring)
        for _ in offspring:
            # Swap two random neighbors
            if np.random.uniform() < mutation_rate:
                idx1 = int(np.random.uniform(n_cities))
                idx2 = (idx1+1) % n_cities
                offspring = self._swap(offspring, idx1, idx2)
        return offspring

    def calc_distance(self, individual) -> float:

        total_dst = 0.0
        dist_mat = self.dist_mat
        for idx in range(len(individual)-1):
            i = individual[idx]
            j = individual[idx + 1]
            total_dst += dist_mat[i, j]

        # From the last city back home
        total_dst += dist_mat[individual[-1], individual[0]]

        return total_dst

    def eval_cost(self, population):
        costs = []
        for individual in population:
            costs.append(
                self.calc_distance(individual)
            )
        return costs

    def cost2fitness(self, costs):
        """
        Convert cost function to fitness function
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

    def rank_fitness(self, fitness):
        """
        Prepare fitness function for Rank Selection
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

    def _pick_one(self, population: list, ranks: list):
        idx = 0
        r = np.random.uniform()
        while r > 0:
            r -= ranks[idx]
            idx += 1
        return population[idx - 1]

    def select_parent(self, parent, population, fitness) -> list:
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

    def plot_optimal_route(self, ax, permutation):
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

    def plot_individual(self, ax, individual, permutation):
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

    def plot_cities(self, ax):
        cities = self.cities
        # Scatter plot the cities (with labels)
        ax.scatter(cities[:, 0], cities[:, 1], c='b', label='Cities')
        for i, city in enumerate(cities):
            ax.text(city[0], city[1], str(i),
                    fontsize=12, ha='center', va='bottom')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    def run(self,
            n_individuals: int = 10,
            n_generations: int = 200,
            dimension: int = 5,
            max_mutation_rate=0.99,
            min_mutation_rate=0.05,
            alpha=0.95):

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


if __name__ == '__main__':
    np.random.seed(42)
    ga = GeneticAlgorithm()
    # ga.run(dimension=10, n_individuals=10, n_generations=100)
    ga.run(dimension=12, n_individuals=10, n_generations=125)
