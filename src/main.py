from function import Function
from graph import Graph
from solution import BlindSearch

FUNCTIONS = ['sphere',
             'schwefel',
             'rosenbrock',
             'rastrigin',
             'griewank',
             'levy',
             'michalewicz',
             'zakharov',
             'ackley']
STEP = 0.5


def plot_my_functions() -> None:
    for fun in FUNCTIONS:
        Graph(interval=Function.get_interval(fun, STEP),
              fun=Function.get(fun)).plot(str.capitalize(fun))


def animate_my_function(function: str = 'ackley') -> None:
    ackley_graph = Graph(Function.get_interval(function, STEP),
                         Function.get(function))
    interval = Function.get_interval(function, 0.5)
    optimizer = BlindSearch(*interval.bounds, Function.get(function))
    points = optimizer.search(100)
    anim = ackley_graph.animate_points(str.capitalize(function), points)

    anim.save(f'{function}.mp4',
              fps=5,
              extra_args=['-vcodec', 'libx264'],
              progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'))


if __name__ == "__main__":
    plot_my_functions()
    animate_my_function()
