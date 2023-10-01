import numpy as np
from typing import Union, Callable
from interval import Interval


class Function:

    @staticmethod
    def get_interval(function: str, step) -> Interval:
        """
        Get the interval (domain) for a specified mathematical function.

        Parameters:
        - function (str): The name of the mathematical function.
        - step: The step size for the interval.

        Returns:
        - Interval: An Interval object representing the domain of the function.

        Supported functions and their intervals:
        - 'sphere': [0, π, π/100]
        - 'schwefel': [-500, 500, 5]
        - 'rosenbrock': [-10, 10, 0.05]  # [-2.048, 2.048]
        - 'rastrigin': [-5.12, 5.12, 0.1]
        - 'griewank':  [-10, 10, 0.1]  # [-600, 600]
        - 'levy': [-10, 10, 0.1]
        - 'michalewicz': [0, π, π/100]
        - 'zakharov': [-10, 10, 0.1]
        - 'ackley': [-3, 3, 0.1]  # [-32.768, 32.768]

        Raises:
        - ValueError: If the requested function is not found in the supported functions.
        """
        functions = {
            'sphere': [0, np.pi, np.pi/100],
            'schwefel': [-500, 500, 5],
            'rosenbrock': [-10, 10, 0.05],  # [-2.048, 2.048]
            'rastrigin': [-5.12, 5.12, 0.1],
            'griewank':  [-10, 10, 0.1],  # [-600, 600]
            'levy': [-10, 10, 0.1],
            'michalewicz': [0, np.pi, np.pi/100],
            'zakharov': [-10, 10, 0.1],
            'ackley': [-3, 3, 0.1],  # [-32.768, 32.768]
        }
        if function in functions:
            lb, ub, step = functions[function]
            return Interval(lb, ub, step)
        else:
            raise ValueError(f"Function '{function}' not found.")

    @staticmethod
    def get(function_name: str) -> Callable:
        """
        Get a callable mathematical function by name.

        Parameters:
        - function_name (str): The name of the mathematical function to retrieve.

        Returns:
        - Callable: A callable function that takes an array-like input and computes the function's value.

        Supported function names:
        - 'sphere': Sphere Function
        - 'schwefel': Schwefel Function
        - 'rosenbrock': Rosenbrock Function
        - 'rastrigin': Rastrigin Function
        - 'griewank': Griewank Function
        - 'levy': Levy Function
        - 'michalewicz': Michalewicz Function
        - 'zakharov': Zakharov Function
        - 'ackley': Ackley Function
        - see https://www.sfu.ca/~ssurjano/optimization.html for details

        Raises:
        - ValueError: If the requested function is not found or not callable.
        """
        function = getattr(Function, str.lower(function_name), None)
        if function is not None and callable(function):
            return lambda x: function(x)
        else:
            raise ValueError(
                f"Function '{function_name}' not found or not callable.")

    @staticmethod
    def sphere(xx: Union[np.ndarray, list]) -> float:
        return np.sum(np.power(xx, 2))

    @staticmethod
    def schwefel(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/schwef.html
        d = len(xx)
        fx = 418.9829 * d - \
            np.sum([x * np.sin(np.sqrt(np.abs(x))) for x in xx])
        return fx

    @staticmethod
    def rosenbrock(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/rosen.html
        d = len(xx)
        fx = np.sum([100 * (xx[i+1] - xx[i]**2)**2 +
                    (xx[i] - 1)**2 for i in range(d-1)])
        return fx

    @staticmethod
    def rastrigin(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/rastr.html
        d = len(xx)
        fx = 10 * d + np.sum([x**2 - 10 * np.cos(2 * np.pi * x) for x in xx])
        return fx

    @staticmethod
    def griewank(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/griewank.html
        d = len(xx)
        sum1 = np.sum([x**2 for x in xx]) / 4000
        prod = np.prod([np.cos(x / np.sqrt(i+1)) for i, x in enumerate(xx)])
        fx = sum1 - prod + 1
        return fx

    @staticmethod
    def levy(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/levy.html
        d = len(xx)
        w = 1 + (xx - 1) / 4
        term1 = (np.sin(np.pi * w[0]))**2
        term2 = np.sum(
            [(w[i] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2) for i in range(d-1)])
        term3 = (w[d-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[d-1]))**2)
        fx = term1 + term2 + term3
        return fx

    @staticmethod
    def michalewicz(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/michal.html
        """
        The Michalewicz function has d! local minima, and it is multimodal.
        The parameter m defines the steepness of they valleys and ridges;
        a larger m leads to a more difficult search.
        The recommended value of m is m = 10.
        """
        d = len(xx)
        m = 10  # recommended value
        fx = -np.sum([np.sin(xx[i]) * (np.sin((i+1) * xx[i]
                     ** 2 / np.pi)**(2*m)) for i in range(d)])
        return fx

    @staticmethod
    def zakharov(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/zakharov.html
        d = len(xx)
        term1 = np.sum([x**2 for x in xx])
        term2 = np.sum([0.5 * (i+1) * xx[i] for i in range(d)])**2
        term3 = np.sum([0.5 * (i+1) * xx[i] for i in range(d)])**4
        fx = term1 + term2 + term3
        return fx

    @staticmethod
    def ackley(xx: Union[np.ndarray, list]) -> float:
        # https://www.sfu.ca/~ssurjano/ackley.html
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(xx)
        term1 = -a * np.exp(-b * np.sqrt(1 / d * np.sum(np.power(xx, 2))))
        term2 = np.exp(1 / d * np.sum([np.cos(c * x) for x in xx]))
        fx = term1 - term2 + a + np.exp(1)
        return fx
