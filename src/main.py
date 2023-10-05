from function import F
from solution import Opt
from graph import plot_my_functions, animate_optimizer, plot_optimizer
import random
import numpy as np


def vizualization_examples():
    plot_my_functions()
    animate_optimizer(F.sphere, Opt.BlindSearch, [15])
    animate_optimizer(F.ackley, Opt.HillClimber, [25, 2], "gif")
    animate_optimizer(F.ackley, Opt.HillClimber, [25, 2], "mp4")
    animate_optimizer(F.michalewicz, Opt.HillClimber, [50,2]) # too low sigma, gets stuck in a valley
    animate_optimizer(F.sphere, Opt.HillClimber, [20,2])
    animate_optimizer(F.ackley, Opt.SimulatedAnnealing, [], "mp4")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    #animate_optimizer(F.ackley, Opt.SimulatedAnnealing, [], "mp4")
    for fun in F:
        plot_optimizer(fun, Opt.SimulatedAnnealing)

