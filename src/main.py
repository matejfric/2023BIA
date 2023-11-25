from function import F
from solution import Opt
from graph import plot_my_functions, animate_optimizer, plot_optimizer, plot_optimizer_contour, animate_optimizer_contour
import random
import numpy as np
import logging

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to the desired level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def vizualization_examples():
    plot_my_functions()
    animate_optimizer(F.sphere, Opt.BlindSearch, [15])
    animate_optimizer(F.ackley, Opt.HillClimber, [25, 2], "gif")
    animate_optimizer(F.ackley, Opt.HillClimber, [25, 2], "mp4")
    animate_optimizer(F.michalewicz, Opt.HillClimber, [50,2]) # too low sigma, gets stuck in a valley
    animate_optimizer(F.sphere, Opt.HillClimber, [20,2])
    animate_optimizer(F.ackley, Opt.SimulatedAnnealing, format="mp4")

    # Pass keyword arguments to the optimizer
    optimizer_args = {'t0': 1000, 't_min': 0, 'alpha': 0.92}
    plot_optimizer(F.griewank, Opt.SimulatedAnnealing, optimizer_args)

    # Loop over all available test functions and
    # plot results of the selected optimizer.
    optimizer_args = {"n_generations": 20, "population_size": 10}
    for fun in F:
        plot_optimizer(fun, Opt.DifferentialEvolution, optimizer_args)

    # Differential Evolution
    optimizer_args = {"n_generations": 20, "population_size": 10}
    animate_optimizer(F.ackley, Opt.DifferentialEvolution, optimizer_args)

    # DE
    optimizer_args = {"n_generations": 40, "population_size": 20}
    animate_optimizer(F.eggholder, Opt.DifferentialEvolution, optimizer_args)

    # PSO
    optimizer_args = {'n_migrations' : 50, 'swarm_size': 20, 'c1':2, 'c2':2}
    plot_optimizer_contour(F.ackley, Opt.ParticleSwarm, optimizer_args)
    animate_optimizer_contour(F.rastrigin, Opt.ParticleSwarm, optimizer_args)

    # Firefly
    plot_optimizer(F.sphere, Opt.Firefly)


def soma_demo():
    optimizer_args = {'n_migrations': 100,
                      'population_size': 20,
                      'perturberation': 0.4,
                      'step': 0.11,
                      'path_length': 3}
    for fun in F:
        plot_optimizer(fun, Opt.SOMA, optimizer_args)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    for fun in F:
        plot_optimizer(fun, Opt.TLBO)
    #plot_optimizer(F.sphere, Opt.Firefly)
    #animate_optimizer(F.sphere, Opt.TLBO, format='gif')    

