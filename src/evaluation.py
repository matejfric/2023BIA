import numpy as np
from typing import Callable, Union
import logging
from function import Function, F
from solution import Optimizer, Opt
import random
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm

def evaluate_optimizer(function: F,
                       optimizer: Opt,
                       optimizer_args: Union[list, dict] = [],
                       dimension: int = 30,
                       max_ofe: int = 3000) -> float:
    fun = Function.get(function)
    interval = Function.get_interval(function)
    optimizer_algorithm: Optimizer = Optimizer.factory(optimizer, interval, fun)
    optimizer_algorithm.d = dimension
    optimizer_algorithm.max_ofe = max_ofe

    # Process 'optimizer_args'
    if isinstance(optimizer_args, list):
        points = optimizer_algorithm.run(*optimizer_args)
    elif isinstance(optimizer_args, dict):
        try:
            points = optimizer_algorithm.run(**optimizer_args)
        except TypeError as e:
            points = optimizer_algorithm.run()
            logging.error(f"TypeError: {e}\nContinuing with default parameters...")
    else:
        points = optimizer_algorithm.run()
        logging.error(f"""'optimizer_args' must be a list or a dictionary.
                      Provided 'optimizer_args' is '{type(optimizer_args)}'
                      with value '{optimizer_args}'.\n
                      Continuing with default parameters...""")
    return optimizer_algorithm.fx

def evaluation_demo(n_experiments: int = 30,
                    dimension: int = 30,
                    population_size: int = 30,
                    max_ofe: int = 3_000):
    max_iter = 1_000_000
    optimizers = {
        Opt.DifferentialEvolution: {"n_generations": max_iter,
                                    "population_size": population_size},
        Opt.ParticleSwarm: {'n_migrations' : max_iter,
                            'swarm_size': population_size,
                            'c1':2, 'c2':2},
        Opt.SOMA: {'n_migrations': max_iter,
                    'population_size': population_size,
                    'perturberation': 0.4,
                    'step': 0.11,
                    'path_length': 3},
        Opt.Firefly: [],
        Opt.TLBO: []
    }

    dataframes = {}
    results_function = {}
    for fun in tqdm(F):
        if F.eggholder is fun and dimension > 2:
            # Eggholder is defined for 2 dimensions only.
            continue
        for opt, opt_args in optimizers.items():
            experiment = np.zeros(n_experiments, dtype=np.float32)
            for i in range(n_experiments):
                fx = evaluate_optimizer(fun, opt, opt_args, dimension, max_ofe)
                experiment[i] = fx
            results_function[opt.name] = list(experiment) +\
                  [np.mean(experiment), np.std(experiment), np.min(experiment)]
        columns = ["Experiment"] + [o.name for o in optimizers.keys()]
        exp_col = list(np.arange(n_experiments)) + ['mean', 'std', 'min']
        column_data = [exp_col] + [v for v in results_function.values()]
        dataframes[str.capitalize(fun.name)] = pd.DataFrame({c: d for c,d in zip(columns, column_data)})

    # Specify the Excel file path
    excel_file_path = 'evaluation2.xlsx'

    # Save each DataFrame to a separate sheet in the Excel file
    with pd.ExcelWriter(excel_file_path) as writer:
        for sheet_name, df in dataframes.items():
            df: pd.DataFrame
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'Excel file "{excel_file_path}" created successfully.')
            

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    evaluation_demo()  
    #evaluation_demo(2,5,5,200)  
