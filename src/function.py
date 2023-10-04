from typing import Union, Callable
import numpy as np
from interval import Interval
from aenum import (Enum,
                   NoAlias)  # solves the issue with non-unique values of the Enum members


class F(Enum, settings=NoAlias):
    """
    Enumeration of functions and their corresponding
    itervals in format [lower_bound, upper_bound, step].
    """
    sphere = [-5, 5, 0.1]
    schwefel = [-500, 500, 5]
    rosenbrock = [-10, 10, 0.05]  # [-2.048, 2.048]
    rastrigin = [-5.12, 5.12, 0.1]
    griewank = [-10, 10, 0.1]  # [-600, 600]
    levy = [-10, 10, 0.1]
    michalewicz = [0, np.pi, np.pi/100]
    zakharov = [-10, 10, 0.1]
    ackley = [-3, 3, 0.05]

    def __iter__(self):
        """
        Example: 
        ```
        for f in F:  
          print(f)
          print(f"{f.name} {f.value})
        ```
        """
        return iter(list(F))


class Function:
    @staticmethod
    def get_interval(function: F) -> Interval:
        """
        Get an interval (lower bound, upper bound, and step) for the given function.

        Args:
            function (F): An enumeration object of type 'F' for which to retrieve the interval.

        Returns:
            Interval: An Interval object representing the lower bound, upper bound,
                      and step size of the function.

        Raises:
            ValueError: If the input 'function' is not an instance of 'F'.
        """
        if isinstance(function, F):
            lb, ub, step = function.value
            return Interval(lb, ub, step)
        else:
            raise ValueError(f"Function '{function}' not found.")

    @staticmethod
    def get(function: F) -> Callable:
        """
        Get a callable function for the given function name.

        Args:
            function (F): An enumeration object of type 'F'.

        Returns:
            Callable: A callable function that corresponds to the provided 'function'.

        Raises:
            ValueError: If the 'function' is not found or is not callable.
        """
        fun = getattr(Function, str.lower(function.name), None)
        if fun is not None and callable(fun):
            return lambda x: fun(x)
        else:
            raise ValueError(
                f"Function '{function.name}' not found or not callable.")

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
