from typing import List, Callable
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from point import Point
from interval import Interval


class Opt(Enum):
    """
    Enumeration of optimization algorithms.
    """
    BlindSearch = auto()
    HillClimber = auto()
    SimulatedAnnealing = auto()


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    """
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        """
        Initialize an optimizer instance.

        Args:
            lower_bound (float): The lower bound of the search space.
            upper_bound (float): The upper bound of the search space.
            objective_function (Callable): The objective function to be optimized.
            dimension (int, optional): The dimensionality of the search space (default is 2).
        """
        self.d = dimension
        self.lb = lower_bound  # we will use the same bounds for all parameters
        self.ub = upper_bound
        self.params = np.zeros(self.d)  # solution parameters
        self.fx = np.inf  # objective function evaluation
        self.objective_function = objective_function

    @staticmethod
    def factory(optimizer: Opt, interval: Interval, function: Callable):
        """
        Factory method to create an instance of an optimizer based on the provided enum.

        Args:
            optimizer (Opt): The optimizer enum value.
            interval (Interval): The interval containing lower and upper bounds for the optimizer.
            function (Callable): The objective function to be optimized.

        Returns:
            Optimizer or None: An instance of the selected optimizer or None if the optimizer is not found.
        """
        cls = get_class(optimizer.name)
        if cls is None:
            print(f"Optimizer '{optimizer.name}' not found.")
            return None
        instance = cls(*interval.bounds, function)
        return instance

    @abstractmethod
    def run(self, *args, **kwargs) -> list[Point]:
        """
        This abstract method allows different arguments in subclasses of Optimizer.
        """
        pass


class BlindSearch(Optimizer):
    """
    BlindSearch optimizer for randomly exploring the search space to find the optimal solution.

    Args:
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        objective_function (Callable): The objective function to be optimized.
        dimension (int, optional): The dimensionality of the search space (default is 2).
    """
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        super().__init__(lower_bound, upper_bound, objective_function, dimension)

    def run(self, n_generations: int = 100) -> list[Point]:
        """
        Run the BlindSearch optimizer to randomly explore the search space and find the optimal solution.

        Args:
            n_generations (int, optional): The number of generations to run the optimizer (default is 100).

        Returns:
            list[Point]: A list of Point objects representing the points generated during optimization.
        """
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
    """
    HillClimber optimizer for finding the optimal solution within a bounded search space.

    Args:
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        objective_function (Callable): The objective function to be optimized.
        dimension (int, optional): The dimensionality of the search space (default is 2).
    """
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        super().__init__(lower_bound, upper_bound, objective_function, dimension)
        # Initial guess (starting point)
        self.params = np.random.uniform(self.lb, self.ub, self.d)

    def generate_individual(self) -> np.ndarray:
        """
        Generate a random individual within the search space.

        Returns:
            np.ndarray: A random individual within the search space.
        """
        # Set sigma to a reasonable amount
        sigma = (self.ub - self.lb) * 3 / 42
        while True:
            individual = np.random.normal(self.params, scale=sigma)
            # Ensure that the generated random solution is within the search space
            if np.all(self.lb <= individual) and np.all(individual <= self.ub):
                return individual

    def run(self, n_generations: int = 100, n_neighbors: int = 1) -> list[Point]:
        """
        Run the HillClimber optimizer to find the optimal solution.

        Args:
            n_generations (int, optional): The number of generations to run the optimizer (default is 100).
            n_neighbors (int, optional): The number of neighbors to consider for each generation (default is 1).

        Returns:
            list[Point]: A list of Point objects representing the points generated during optimization.
        """
        points = []

        for _ in range(n_generations):
            individuals = []
            for _ in range(n_neighbors):
                # Generate a random solution within the search space
                individuals = [self.generate_individual() for _ in range(n_neighbors)]

            # Select the best individual
            for individual in individuals:
                fx = self.objective_function(individual)
                if fx < self.fx:
                    self.params = individual
                    self.fx = fx
                # Append all generated individuals
                points.append(Point(individual[0], individual[1], fx))

        return points


class SimulatedAnnealing(Optimizer):
    """
    SimulatedAnnealing optimizer for finding the optimal solution within a bounded search space.

    Args:
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        objective_function (Callable): The objective function to be optimized.
        dimension (int, optional): The dimensionality of the search space (default is 2).

    Simulated annealing is a stochastic optimization technique inspired by metallurgy principles
    that involve gradually cooling a material to achieve a more optimal state,
    akin to "Strike while the iron is hot."
    """
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        super().__init__(lower_bound, upper_bound, objective_function, dimension)
        # Initial guess (starting point)
        self.params = np.random.uniform(self.lb, self.ub, self.d)

    def generate_individual(self) -> np.ndarray:
        """
        Generate a random individual within the search space.

        Returns:
            np.ndarray: A random individual within the search space.
        """
        # Set sigma to a reasonable amount
        sigma = (self.ub - self.lb) * 3 / 42
        while True:
            individual = np.random.normal(self.params, scale=sigma)
            # Ensure that the generated random solution is within the search space
            if np.all(self.lb <= individual) and np.all(individual <= self.ub):
                return individual

    def run(self, t0: int = 1000, t_min: int = 0, step: int = 10) -> list[Point]:        
        """
        Run Simulated annealing optimizer.

        Parameters:
            t0 (int, optional): Initial temperature (default is 1000).
            t_min (int, optional): Minimum temperature at which the optimization stops (default is 0).
            step (int, optional): Temperature reduction step (default is 10).

        Returns:
            list[Point]: A list of Point objects representing the points generated during optimization.
        """
        t = t0
        self.fx = self.objective_function(self.params)
        points = []

        while t > t_min:
            individual = self.generate_individual()
            fx = self.objective_function(individual)
            points.append(Point(individual[0], individual[1], fx))

            delta_fx = fx - self.fx
            # Update solution if the new solution is better 
            if delta_fx < 0:
                self.params = individual
            else:
                # Generate a random number between 0 (inclusive) and 1 (exclusive)
                r = np.random.uniform(0,1)
                # e^(\delta_f / t) ...if 't' is large, this expression is close to zero;
                # therefore, there's high probability to accept even the worse solution.
                # "Strike while the iron is hot."
                # As 't' gets smaller, this chance becomes smaller and this algortihm
                # acts just like HillClimber.
                if r < np.exp(- delta_fx / t):
                    self.params = individual
            t -= step

        return points


def get_class(class_name: str):
    """
    Retrieve a class object by its name from the global namespace and return an instance of the class.

    Args:
        class_name (str): The name of the class to retrieve.

    Returns:
        type or None: The class object if found, or None if the class is not found in the global namespace.
    """
    try:
        # Use globals() to get the class by name and create an instance
        cls = globals()[class_name]
        return cls
    except KeyError:
        print(f"Class '{class_name}' not found")
        return None
