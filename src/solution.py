from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from point import Point
from interval import Interval
from scipy.spatial.distance import euclidean

class Opt(Enum):
    """
    Enumeration of optimization algorithms.
    """
    BlindSearch = auto()
    HillClimber = auto()
    SimulatedAnnealing = auto()
    DifferentialEvolution = auto()
    ParticleSwarm = auto()
    SOMA = auto()
    Firefly = auto()


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
                individuals = [self.generate_individual()
                               for _ in range(n_neighbors)]

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

    def run(self, t0: int = 1000, t_min: int = 5, alpha: float = 0.93) -> list[Point]:
        """
        Run Simulated annealing optimizer.

        Parameters:
            t0 (int, optional): Initial temperature (default is 1000).
            t_min (int, optional): Minimum temperature at which the optimization stops (default is 0).
            alpha (int, optional): Temperature reduction alpha (default is 10).

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
                r = np.random.uniform(0, 1)
                # e^(\delta_f / t) ...if 't' is large, this expression is close to zero;
                # therefore, there's high probability to accept even the worse solution.
                # "Strike while the iron is hot."
                # As 't' gets smaller, this chance becomes smaller and this algortihm
                # acts just like HillClimber.
                if r < np.exp(- delta_fx / t):
                    self.params = individual
            t *= alpha

        return points


class DifferentialEvolution(Optimizer):
    """
    Differential Evolution optimizer for solving optimization problems.

    Parameters:
    - lower_bound (float): The lower bound for the optimization problem.
    - upper_bound (float): The upper bound for the optimization problem.
    - objective_function (Callable): The objective function to be minimized.
    - dimension (int): The dimension of the problem (default is 2).
    """

    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        """
        Initialize the DifferentialEvolution optimizer.

        Args:
        - lower_bound (float): The lower bound for the optimization problem.
        - upper_bound (float): The upper bound for the optimization problem.
        - objective_function (Callable): The objective function to be minimized.
        - dimension (int): The dimension of the problem (default is 2).
        """
        super().__init__(lower_bound, upper_bound, objective_function, dimension)

    def _generate_initial_population(self, population_size: int) -> list[np.ndarray]:
        """
        Generate an initial population of random solutions.

        Args:
        - population_size (int): The number of individuals in the population.

        Returns:
        - initial_population (np.ndarray): An array of random solutions.
        """
        initial_population = [np.random.uniform(self.lb, self.ub, self.d)
                              for _ in range(population_size)]
        return initial_population

    def _evaluate_individual(self, individual: list[float]) -> float:
        """
        Evaluate the fitness of an individual solution using the objective function.

        Args:
        - individual (list[float]): The parameters of an individual solution.

        Returns:
        - fitness (float): The fitness value of the individual.
        """
        return self.objective_function(individual)

    def _select_random_parent_indices(self, idx: int, population_size: int) -> np.ndarray:
        """
        Select three random parent indices from the population for mutation.

        Args:
        - idx (int): The index of the current individual.
        - population_size (int): The total size of the population.

        Returns:
        - selected_indices (np.ndarray): An array of three randomly chosen parent indices.
        """
        indices = np.arange(population_size)
        # Remove the 'idx' element (so that it isn't chosen)
        indices = indices[indices != idx]
        # Choose three random parent indices from the population
        selected_indices = np.random.choice(indices, 3, replace=False)
        return selected_indices

    def _calc_mutation_vector(self, population: list[list[float]], idxs: list[int], mutation_constant: float) -> list[int]:
        """
        Calculate the mutation vector for an individual. 
        Uses "movement on the sphere" to ensure that
        the solution stays inside the feasible set.

        Args:
        - population (list[list[float]): The population of individuals.
        - idxs (list[int]): Indices of three parent individuals.
        - mutation_constant (float): The mutation constant.

        Returns:
        - mutation_vector (list[int]): The mutation vector for the individual.
        """
        vec = (population[idxs[0]] - population[idxs[1]]) * \
            mutation_constant + population[idxs[2]]
        # Ensure that the vector stays inside the feasible set (přípustná množina)
        for i in range(len(vec)):
            # "Movement on the sphere"
            if vec[i] < self.lb:
                diff = abs(self.lb - vec[i])
                vec[i] = self.ub - diff
            elif vec[i] > self.ub:
                diff = abs(self.ub - vec[i])
                vec[i] = self.lb + diff
        return vec

    def _crossover(self, mutation_vec: list[float], parent: list[float], crossover_rate: float) -> list[float]:
        """
        Perform crossover between an individual and the mutation vector to generate a trial vector.

        Args:
        - mutation_vec (list[float]): The mutation vector.
        - parent (list[float]): The parent individual.
        - crossover_rate (float): The crossover rate.

        Returns:
        - trial_vector (list[float]): The trial vector generated by crossover.
        """
        trial_vector = np.zeros(self.d)
        rnd = np.random.randint(0, self.d)
        for idx in range(self.d):
            if np.random.uniform() < crossover_rate or idx == rnd:
                # At least 1 parameter should be from the mutation vector
                trial_vector[idx] = mutation_vec[idx]
            else:
                # Copy parameters from parent
                trial_vector[idx] = parent[idx]
        return trial_vector

    def run(self, n_generations: int = 100, population_size: int = None, mutation_constant: float = 0.8, crossover_rate: float = 0.5) -> list[Point]:
        """
        Run the Differential Evolution optimization algorithm for a specified number of generations.

        Args:
        - n_generations (int): The number of generations to run the optimization.
        - population_size (int): The size of the population.
        - mutation_constant (float): The mutation constant.
        - crossover_rate (float): The crossover rate.

        Returns:
        - visited_solutions (list[Point]): A list of Point objects representing visited solutions during optimization.
        """
        NP = 10 * self.d if population_size == None else population_size
        F = mutation_constant
        CR = crossover_rate

        population = self._generate_initial_population(NP)
        visited_solutions = []

        # Evaluate fitness of the initial population.
        # Keep in mind that we intend to minimize 
        # the number of objective evaluations.
        fittness = [self._evaluate_individual(ind) for ind in population]

        for _ in range(n_generations):
            new_population = population.copy()
            new_fitness = fittness.copy()
            for i, parent in enumerate(population):
                idxs = self._select_random_parent_indices(i, NP)
                mutation_vec = self._calc_mutation_vector(population, idxs, F)
                trial_vector = self._crossover(mutation_vec, parent, CR)

                # Evaluate fittness
                fittness_offspring = self._evaluate_individual(trial_vector)
                fittness_parent = fittness[i] # Here, we save redundant objective 
                                              # evaluations by reusing previously 
                                              # computed fitness.

                # Solution with the same fittness as a target vector is always accepted
                if fittness_offspring <= fittness_parent:

                    # Propagate the offspring
                    new_population[i] = trial_vector
                    new_fitness[i] = fittness_offspring

                    if fittness_offspring <= self.fx:
                        # Update the best solution
                        self.fx = fittness_offspring
                        self.params = trial_vector

                    visited_solutions.append(
                        Point(self.params[0], self.params[1], self.fx))

            population = new_population
            fittness = new_fitness

        return visited_solutions


@dataclass
class Particle:
    x: np.ndarray # position
    fx: float # fitness(x)
    v: np.ndarray # velocity
    p_best: np.ndarray # best visited position
    f_best: float # fitness(p_best)


class Swarm:
    def __init__(self) -> None:
        self.particles: list[Particle] = []
        self.g_best: int = 0

    def get_g_best(self) ->  Particle:
        return self.particles[self.g_best]

    def append(self, particle: Particle) -> None:
        self.particles.append(particle)

    def __getitem__(self, index: int) -> Particle:
        if index < 0 or index >= len(self.particles):
            raise IndexError(f"Invalid index '{index}'")
        return self.particles[index]
    
    def __setitem__(self, index: int, particle: Particle) -> None:
        if index < 0 or index >= len(self.particles):
            raise IndexError(f"Invalid index '{index}'")
        if not isinstance(particle, Particle):
            raise ValueError(f"Particle must have the data type 'Particle', but has the type '{type(particle)}'")
        self.particles[index] = particle
    
    def __iter__(self) -> Iterable[Particle]:
        return iter(self.particles)


class ParticleSwarm(Optimizer):
    """
    swarm <=> population
    """

    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        """
        Initialize the ParticleSwarm optimizer.

        Args:
        - lower_bound (float): The lower bound for the optimization problem.
        - upper_bound (float): The upper bound for the optimization problem.
        - objective_function (Callable): The objective function to be minimized.
        - dimension (int): The dimension of the problem (default is 2).
        """
        super().__init__(lower_bound, upper_bound, objective_function, dimension)

    def _generate_initial_swarm(self, swarm_size: int, v_min: float, v_max: float) -> Swarm:
        # Initialize the particle's position with a uniformly distributed random vector: x_i ~ U(lb, ub)
        xs = [np.random.uniform(self.lb, self.ub, self.d)
                              for _ in range(swarm_size)] 
        
        # Initialize the particle's velocity: v_i ~ U(-|ub-lb|, |ub-lb|)
        vs = [np.random.uniform(v_min, v_max, self.d)
                              for _ in range(swarm_size)]
        
        # Initialize the particle's velocity: v_i ~ N((v_max+v_min)/2,abs(v_max-v_min)/4)
        # vs = [np.random.normal((v_max+v_min)/2,abs(v_max-v_min)/4,self.d)
        #                       for _ in range(swarm_size)]

        # Initialize the particle's best known position to its initial position: p_i ← x_i
        # ps = xs

        # Create initial swarm
        swarm = Swarm()
        best_fitness = np.inf
        for idx,x,v in zip(range(len(xs)),xs,vs):
            f = self.objective_function(x)
            swarm.append(Particle(x,f,v,x,f))
            if f < best_fitness:
                best_fitness = f
                swarm.g_best = idx
        return swarm
    
    def _compute_inertia(self, n_migrations: int, migration: int,
                         initial_inertia: float, final_inertia: float) -> float:
        return initial_inertia - (((initial_inertia - final_inertia) * migration) / n_migrations)
    
    def _check_velocity_boundaries(self, v: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        # for idx in range(len(v)):
        #     if v[idx] < v_min:
        #         v[idx] = v_min
        #     if v[idx] > v_max:
        #         v[idx] = v_max
        # return v
        coefficient = abs(v_max - v_min) / 10
        for idx in range(len(v)):
            if v[idx] < v_min:
                v[idx] = v_min #+ coefficient
            if v[idx] > v_max:
                v[idx] = v_max #- coefficient
        return v
        # for idx in range(len(v)):
        #     if v[idx] < v_min or v[idx] > v_max:
        #         v = np.random.normal((v_max+v_min)/2,abs(v_max-v_min)/4,self.d)
        # return v

    def _check_position_boundaries(self, x: np.ndarray) -> np.ndarray:
        for idx in range(len(x)):
            if x[idx] < self.lb or x[idx] > self.ub:
                return np.random.uniform(self.lb,self.ub,self.d)
        return x
        
    def _update_best_particle_position(self, particle_idx: int, swarm: Swarm) -> Swarm:
        p: Particle = swarm[particle_idx] 
        g_best: Particle = swarm.get_g_best()
        p.fx = self.objective_function(p.x)
        
        if p.fx < p.f_best:
            # Update p_best
            p.p_best = p.x
            p.f_best = p.fx
            swarm[particle_idx] = p

            if p.fx < g_best.f_best:
                # Update g_Best
                swarm.g_best = particle_idx
        return swarm
    
    def _update_swarm_velocity_and_position(self, swarm: Swarm, inertia: float,
                         c1: float, c2: float,
                         v_min: float, v_max: float) -> Swarm:
        g_best = swarm.get_g_best()
        w = inertia
        i: int
        p: Particle
        for i,p in enumerate(swarm):
            rand_p = np.random.uniform(0,1,self.d)
            rand_g = np.random.uniform(0,1,self.d)

            # Update velocity
            p.v = w * p.v + c1 * rand_p * (p.p_best - p.x) + \
                  c2 * rand_g * (g_best.p_best - p.x)
            p.v = self._check_velocity_boundaries(p.v, v_min, v_max)

            # Update position
            p.x += p.v
            if not self._is_within_boundaries(p,v_min, v_max):
                p.x = self._check_position_boundaries(p.x)

            self._assert_boundaries(p,v_min,v_max)

            # Update the particle
            swarm[i] = p

            # Update p_best and g_best
            swarm = self._update_best_particle_position(i, swarm)

        return swarm
    
    def _assert_boundaries(self, particle: Particle, v_min: float, v_max: float) -> None:
        for idx in range(len(particle.x)):
            if (particle.x[idx] < self.lb or particle.x[idx] > self.ub or
                particle.v[idx] < v_min or particle.v[idx] > v_max):
                raise RuntimeError('Particle out of boundaries')
            
    def _is_within_boundaries(self, particle: Particle, v_min: float, v_max: float) -> bool:
        for idx in range(len(particle.x)):
            if (particle.x[idx] < self.lb or particle.x[idx] > self.ub or
                particle.v[idx] < v_min or particle.v[idx] > v_max):
                return False
        return True
            
    def run(self, 
            n_migrations: int = 50,
            swarm_size: int | None = None,
            v_min: float | None = None,
            v_max: float | None = None,
            c1: float | None = 0.5,
            c2: float | None = 0.5,
            initial_inertia: float = 0.9,
            final_inertia: float = 0.4,
            ) -> list[Point]:
        """

        Returns:
        - visited_solutions (list[Point]): A list of Point objects representing visited solutions during optimization.
        """
        swarm_size = 10 * self.d if swarm_size is None else swarm_size
        v_max = 1 / 20 * abs(self.ub - self.lb) if v_max is None else v_max
        v_min = (-1) * v_max if v_min is None else v_min
        visited_solutions = []

        swarm: Swarm = self._generate_initial_swarm(swarm_size, v_min, v_max)

        for migration in range(n_migrations):
            w = self._compute_inertia(n_migrations,migration,initial_inertia,final_inertia)
            #w = 1
            swarm = self._update_swarm_velocity_and_position(swarm,w,c1,c2,v_min,v_max)

            # Save only g_best
            # particle: Particle = swarm.get_g_best()
            # visited_solutions.append(
            #     Point(particle.p_best[0], particle.p_best[1], particle.f_best))
            for particle in swarm:
                particle: Particle
                self._assert_boundaries(particle, v_min, v_max)
                visited_solutions.append(
                    Point(particle.p_best[0], particle.p_best[1], particle.f_best))

        # Save the best solution
        g_best = swarm.get_g_best()
        self._assert_boundaries(g_best, v_min, v_max)
        self.params = g_best.p_best
        self.fx = g_best.f_best
        visited_solutions.append(Point(self.params[0], self.params[1], self.fx))

        return visited_solutions


class SOMA(Optimizer):
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        """
        Initialize the Self-Organizing Migrating Algorithm (SOMA).

        Args:
        - lower_bound (float): The lower bound for the optimization problem.
        - upper_bound (float): The upper bound for the optimization problem.
        - objective_function (Callable): The objective function to be minimized.
        - dimension (int): The dimension of the problem (default is 2).
        """
        super().__init__(lower_bound, upper_bound, objective_function, dimension)

    def _generate_initial_population(self, population_size: int) -> np.ndarray:
        population = [np.random.uniform(self.lb, self.ub, self.d)
                              for _ in range(population_size)] 
        return population
    
    def _check_boundaries(self, vec: np.ndarray) -> np.ndarray:
        # Ensure that the vector stays inside the feasible set
        for d in range(self.d):
            if vec[d] < self.lb or vec[d] > self.ub:
                # If out of bound reset the respective parameter
                vec[d] = np.random.uniform(self.lb, self.ub)
        return vec
    
    def _go_in_direction_to_leader(self, 
                                   individual: np.ndarray,
                                   leader: np.ndarray, 
                                   step_sizes: list[float],
                                   perturberation: float) -> list[np.ndarray]:
        migration_loop = []
        prt_vec = np.random.uniform(0,1,self.d) < perturberation
        for t in step_sizes:
            one_step = individual + t * prt_vec * (leader - individual)
            one_step = self._check_boundaries(one_step)
            migration_loop.append(one_step)
        return migration_loop
        
    def run(self,
            n_migrations: int = 10,
            population_size: int = 10,
            perturberation: float = 0.5,
            step: float = 0.13,
            path_length: float = 1) -> list[Point]:
        visited_solutions = []
        population = self._generate_initial_population(population_size)
        fitness = [self.objective_function(i) for i in population]
        leader_idx = np.argmin(fitness)
        steps = np.arange(0,path_length,step)

        for _ in range(n_migrations):
            leader = population[leader_idx]
            new_population = population.copy()
            for i in range(population_size):
                # Leader stays put
                if i == leader_idx:
                    continue
                individual = population[i]
                migration_loop = self._go_in_direction_to_leader(
                    individual, leader, steps, perturberation
                )
                # Rank the individual
                ind_fitness = [self.objective_function(m) 
                               for m in migration_loop]
                # Update best position
                current_best_pos_idx = np.argmin(ind_fitness)
                current_best_fitness = ind_fitness[current_best_pos_idx]
                if current_best_fitness < fitness[i]:
                    fitness[i] = current_best_fitness
                    new_population[i] = migration_loop[current_best_pos_idx]
                    if current_best_fitness < fitness[leader_idx]:
                        leader_idx = i
                        visited_solutions.append(
                            Point(new_population[i][0], new_population[i][1], fitness[i]))
            # Update leader
            leader_idx = np.argmin(fitness)
            # Update population
            population = new_population

        # Save the best solution
        self.params = population[leader_idx]
        self.fx = fitness[leader_idx]
        visited_solutions.append(
            Point(self.params[0], self.params[1], self.fx))
        return visited_solutions
    

class Firefly(Optimizer):
    def __init__(self, lower_bound: float, upper_bound: float, objective_function: Callable, dimension: int = 2):
        """
        Initialize the Self-Organizing Migrating Algorithm (SOMA).

        Args:
        - lower_bound (float): The lower bound for the optimization problem.
        - upper_bound (float): The upper bound for the optimization problem.
        - objective_function (Callable): The objective function to be minimized.
        - dimension (int): The dimension of the problem (default is 2).
        """
        super().__init__(lower_bound, upper_bound, objective_function, dimension)

    def _generate_initial_population(self, population_size: int) -> np.ndarray:
        population = [np.random.uniform(self.lb, self.ub, self.d)
                      for _ in range(population_size)] 
        return population
    
    def _check_boundaries(self, vec: np.ndarray) -> np.ndarray:
        """Ensure that the vector stays inside the feasible set"""
        for d in range(self.d):
            if vec[d] < self.lb or vec[d] > self.ub:
                # If out of bound reset the respective parameter
                vec[d] = np.random.uniform(self.lb, self.ub)
        return vec
    
    def _compute_attractiveness(self,
                                ffi: np.ndarray,
                                ffj: np.ndarray,
                                initial_attractivness: float,
                                light_absorption: float | None) -> float:
        # Compute distance between firefly i and j 
        r = euclidean(ffi, ffj)
        # Compute attractiveness between firefly i and j 
        if light_absorption is None:
            attractiveness = 1 / (1+r)
        else:
            attractiveness = initial_attractivness * \
                np.exp(-light_absorption * r**2)
        return attractiveness

    def run(self,
            n_generations: int = 10,
            population_size: int = 10,  # number of fireflys
            alpha: float = 0.5,
            light_absorption: float | None = None,
            initial_attractivness: float = 1,
            step: float = 0.13,
            path_length: float = 1) -> list[Point]:
        
        def f(x): return self.objective_function(x)

        visited_solutions = []
        population = self._generate_initial_population(population_size)
        light_intensity = [f(i) for i in population]  # fitness
        best_ff_idx = np.argmin(light_intensity)

        for _ in range(n_generations):
            for i in range(population_size):
                for j in range(i):
                    if light_intensity[j] > light_intensity[i]:
                        ffi = population[i]  # i-th firefly
                        ffj = population[j]  # j-th firefly

                        attractiveness = self._compute_attractiveness(
                            ffi, ffj, initial_attractivness, light_absorption)
                        eps = np.random.normal(size=self.d)

                        # Move firefly i towards j                
                        ffi += attractiveness * (ffj - ffi) + alpha * eps
                        ffi = self._check_boundaries(ffi)

                        # Evaluate solution and update light intesity
                        f_new = f(ffi)
                        if f_new <= light_intensity[i]:
                            # Update local solution
                            population[i] = ffi
                            light_intensity[i] = f_new

                            visited_solutions.append(
                                    Point(ffi[0],
                                          ffi[1],
                                          f_new)
                                          )

                            if f_new <= light_intensity[best_ff_idx]:
                                # Update global solution
                                best_ff_idx = i
                                self.params = ffi
                                self.fx = f_new

                        if i != best_ff_idx:
                            # Other fireflys always move
                            population[i] = ffi
                            light_intensity[i] = f_new
        return visited_solutions


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
