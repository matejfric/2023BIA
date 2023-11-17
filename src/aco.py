# 'aco.py'
# Ant Colony Optimization

import logging
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from tsp import TSP
from tsp_plotting import plot_individual


class AntColonyOptimization:
    """
    Ant Colony Optimization for finding the optimal solution of TSP.
    """

    def generate_cities(self, n_cities: int) -> np.ndarray:
        """
        Generate city coordinates for the TSP instance.

        Args:
            n_cities (int): The number of cities in the TSP.
        """
        tsp = TSP(n_cities)
        self.cities = tsp.cities

    def generate_distances(self, n_cities: int) -> np.ndarray:
        """
        Generate the distance matrix for the TSP based on city coordinates.

        Args:
            n_cities (int): The number of cities in the TSP.
        """
        tsp = TSP(n_cities)
        self.dist_mat = tsp.dist_mat

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

    def run(self,
            n_iterations: int = 1000,
            n_cities: int = 12,
            n_ants: int = 10,
            pheromone_evaporation: float = 0.05,
            alpha: float = 1,
            beta: float = 1
            ) -> None:
        """
        Run the ACO to solve the TSP and visualize the results.
        """

        # Load cities
        self.generate_cities(n_cities)
        # Precompute distance matrix
        self.generate_distances(n_cities)
        # Prepare plot
        _, ax = plt.subplots(1, 1)
        # Get reference solution for comparison
        _, permutation_opt = TSP(n_cities).solve()

        # Compute vision matrix
        mask = ~np.eye(n_cities, dtype=bool)
        vision_mat = np.reciprocal(self.dist_mat, where=mask)

        # Initialize pheromones
        initial_pheromone_value = 1
        pheromones = initial_pheromone_value + np.zeros((vision_mat.shape))
        rho = pheromone_evaporation

        f_best: float = np.inf
        best_permutation: list = []

        def f(x): return self.calc_distance(x)

        # Repeat for number of iterations
        for _ in tqdm(range(n_iterations), "Ant Colony Optimization"):
            starting_cities = np.random.choice(np.arange(n_cities), n_ants)
            ant_colony = []

            # Path finding
            for ant in range(n_ants):
                local_vision_mask = np.ones((vision_mat.shape), dtype=bool)
                local_permutation = []
                city = starting_cities[ant]
                while True:
                    local_permutation.append(city)
                    local_vision_mask[:, city] = False
                    unvisited_cities = local_vision_mask[city, :]
                    if np.sum(unvisited_cities) == 0:
                        break
                    costs = np.zeros(len(unvisited_cities))
                    costs[unvisited_cities] = pheromones[city, unvisited_cities] ** alpha * \
                        vision_mat[city, unvisited_cities] ** beta
                    denom = np.sum(costs)
                    probabilities = costs / denom
                    city = np.random.choice(
                        len(probabilities), p=probabilities)
                ant_colony.append(local_permutation)

            # Update pheromones and solution
            delta_pheromones = np.zeros_like(pheromones)
            for ant in ant_colony:
                ant_cost = f(ant)

                # Update pheromones
                for i in range(len(ant) - 1):
                    city_i, city_j = ant[i], ant[i + 1]
                    delta_pheromones[city_i, city_j] += 1 / ant_cost

                # Update solution
                if ant_cost < f_best:
                    best_permutation = ant
                    f_best = ant_cost
                    # Visualization
                    plot_individual(ax, ant, permutation_opt, self.cities)

            pheromones = (1 - rho) * pheromones  # Evaporation
            pheromones += delta_pheromones  # Deposit new pheromones

        plot_individual(ax, best_permutation, permutation_opt, self.cities)
        plt.savefig('aco_tsp.png', dpi=300)
        plt.show()
        print(f"ACO:\n{f_best}\n{best_permutation}")


if __name__ == '__main__':
    np.random.seed(42)
    ga = AntColonyOptimization()
    ga.run(n_cities=12)
