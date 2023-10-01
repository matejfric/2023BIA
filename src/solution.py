from typing import List, Callable
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from point import Point
from interval import Interval


class Opt(Enum):
    BlindSearch = auto()
    HillClimber = auto()


class Optimizer(ABC):
    """
  Abstract class
  """

    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        self.d = dimension
        self.lb = lower_bound  # we will use the same bounds for all parameters
        self.ub = upper_bound
        self.params = np.zeros(self.d)  # solution parameters
        self.fx = np.inf  # objective function evaluation
        self.objective_function = objective_function

    @staticmethod
    def factory(optimizer: Opt, interval: Interval, function: Callable):
        cls = get_class(optimizer.name)
        if cls is None:
            print(f"Optimizer '{optimizer.name}' not found.")
            return None
        instance = cls(*interval.bounds, function)
        return instance

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        This abstract method allows different arguments in subclasses of Solution.
      """
        pass


class BlindSearch(Optimizer):
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        super().__init__(lower_bound, upper_bound, objective_function, dimension)

    def run(self, n_generations: int = 100) -> List[np.ndarray]:
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


class HillClimber(Optimizer):
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        super().__init__(lower_bound, upper_bound, objective_function, dimension)
        # Initial guess (starting point)
        self.params = np.random.uniform(self.lb, self.ub, self.d)

    def run(self, n_generations: int = 100, n_neighbors: int = 1) -> List[np.ndarray]:
        points = []
        # Set sigma to a reasonable amount
        sigma = (self.ub - self.lb) * 3 / 42

        for _ in range(n_generations):
            # Generate a random solution within the search space
            individuals = [np.random.normal(self.params, scale=sigma)
                           for _ in range(n_neighbors)]

            # Select the best individual
            for individual in individuals:
                fx = self.objective_function(individual)
                if fx < self.fx:
                    self.params = individual
                    self.fx = fx
                points.append(Point(individual[0], individual[1], fx))

        return points


def get_class(class_name):
    try:
        # Use globals() to get the class by name and create an instance
        cls = globals()[class_name]
        return cls
    except KeyError:
        print(f"Class '{class_name}' not found")
        return None
