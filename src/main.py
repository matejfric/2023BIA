from function import F
from solution import Opt
from graph import plot_my_functions, animate_optimizer, plot_optimizer
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
    animate_optimizer(F.ackley, Opt.SimulatedAnnealing, [], "mp4")

    # Pass keyword arguments to the optimizer
    optimizer_args = {'t0': 1000, 't_min': 0, 'step': 100}
    plot_optimizer(F.griewank, Opt.SimulatedAnnealing, optimizer_args)

    # Loop over all available test functions and
    # plot results of the selected optimizer.
    for fun in F:
        plot_optimizer(fun, Opt.SimulatedAnnealing)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    animate_optimizer(F.griewank, Opt.SimulatedAnnealing, [], "mp4")

