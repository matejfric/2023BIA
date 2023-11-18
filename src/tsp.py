from typing import Any
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics.simulated_annealing import solve_tsp_simulated_annealing


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
        if n_cities > 22:
            self.cities = np.array([np.random.uniform(0, 1000, 2) for _ in range(n_cities)], dtype=np.float64)
            self._dist_mat = None
        elif n_cities < 3:
            raise ValueError("Number of cities must be greater than 2.")
        else:
            distance_matrix, cities = TSP.ulysses(n_cities)
            self.cities = cities
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
    def ulysses(n_cities: int = 4) -> tuple[np.ndarray, np.ndarray[tuple[float]]]:
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
                distance_matrix[i - 1, j - 1] = \
                    euclidean(coordinates[i], coordinates[j])

        # Print the distance matrix
        # for row in distance_matrix:
        #     print(" ".join(map(str, row)))

        cities = np.array(list(coordinates.values()))

        return distance_matrix, cities


if __name__ == '__main__':
    np.random.seed(42)
    tsp = TSP(n_cities=12)
    cities = tsp.cities
    dst_mat = tsp.dist_mat
    distance, permutation = tsp.solve()
    print(f"The optimal solution has a cost of {distance}")
