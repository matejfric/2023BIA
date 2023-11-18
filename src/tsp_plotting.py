import matplotlib.pyplot as plt
import numpy as np


def _plot_optimal_route(ax: plt.Axes, permutation: list, cities: np.ndarray) -> None:
    """
    Plot the optimal TSP route on a given axis.

    Args:
        ax (plt.Axes): The axis on which to visualize the TSP route.
        permutation (list): The optimal TSP tour represented as a permutation of city indices.
    """
    # Plot lines to connect cities in the TSP order
    for i in range(len(permutation) - 1):
        start_city = cities[permutation[i]]
        end_city = cities[permutation[i + 1]]
        ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
                'g', alpha=0.5, linewidth=4)

    # Connect the last city to the first city to complete the TSP loop
    start_city = cities[permutation[-1]]
    end_city = cities[permutation[0]]
    ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
            'g', alpha=0.5, linewidth=4, label='Optimal Route')


def plot_individual(ax: plt.Axes, individual: list, permutation: list, cities: np.ndarray, iter: int = None) -> None:
    """
    Plot an individual's TSP route on a given axis.

    Args:
        ax (plt.Axes): The axis on which to visualize the TSP route.
        individual (list): The TSP solution to be visualized.
        permutation (list): The optimal TSP tour represented as a permutation of city indices.
    """
    # Clear the current axis to remove previous routes
    ax.cla()
    _plot_cities(ax, cities)
    _plot_optimal_route(ax, permutation, cities)

    # Plot lines to connect cities in the TSP order
    for i in range(len(individual) - 1):
        start_city = cities[individual[i]]
        end_city = cities[individual[i + 1]]
        ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
                'r', alpha=0.8, linewidth=3)

    # Connect the last city to the first city to complete the TSP loop
    start_city = cities[individual[-1]]
    end_city = cities[individual[0]]
    ax.plot([start_city[0], end_city[0]], [start_city[1], end_city[1],],
            'r', alpha=0.8, linewidth=3, label='Fittest Individual')

    if iter is None:
        plt.title('TSP Visualization')
    else:
        plt.title(f'TSP Visualization - Iteration #{iter}')
    plt.legend()
    # Pause to display the new plot
    plt.pause(0.01)


def _plot_cities(ax: plt.Axes, cities: np.ndarray) -> None:
    """
    Plot city coordinates on a given axis.

    Args:
        ax (plt.Axes): The axis on which to visualize city coordinates.
    """
    # Scatter plot the cities (with labels)
    ax.scatter(cities[:, 0], cities[:, 1], c='b', label='Cities')
    for i, city in enumerate(cities):
        ax.text(city[0], city[1], str(i),
                fontsize=12, ha='center', va='bottom')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
