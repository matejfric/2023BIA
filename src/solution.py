from typing import List, Callable
import numpy as np

from point import Point


class Solution:
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        self.d = dimension
        self.lb = lower_bound  # we will use the same bounds for all parameters
        self.ub = upper_bound
        self.params = np.zeros(self.d)  # solution parameters
        self.fx = np.inf  # objective function evaluation
        self.objective_function = objective_function


class BlindSearch(Solution):
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        super().__init__(lower_bound, upper_bound, objective_function, dimension)

    def search(self, n_generations: int) -> List[np.ndarray]:
        points = []

        for _ in range(n_generations):
            # Generate a random solution within the search space
            random_point = np.random.uniform(self.lb, self.ub, self.d)

            # Evaluate the objective function for the random solution
            fx = self.objective_function(random_point)
            points.append(Point(random_point[0], random_point[1], fx))

            # Update the best solution if a better solution is found
            if fx < self.fx:
                self.params = random_point
                self.fx = fx
        return points
