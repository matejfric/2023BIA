import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from tsp import TSP
from tsp_plotting import plot_individual


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
            plot_individual(
                ax, population[np.argmin(costs)], permutation_opt, self.cities)

        # Extract solution
        solution_idx = np.argmin(costs)
        permutation = population[solution_idx]
        distance = costs[solution_idx]

        plt.savefig('my_tsp.png', dpi=300)

        print(f"GA:\n{distance}\n{permutation}")


if __name__ == '__main__':
    np.random.seed(42)
    ga = GeneticAlgorithm()
    ga.run(dimension=10, n_individuals=10, n_generations=100)
    # ga.run(dimension=12, n_individuals=10, n_generations=200)
