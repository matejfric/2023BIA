import numpy as np
from typing import Callable, Union
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from interval import Interval
from function import Function, F
from solution import Optimizer, Opt


class Graph:

    def __init__(self, interval: Interval, fun: Callable):
        lb = interval.lb
        ub = interval.ub
        step = interval.step
        self.xs: np.array = np.arange(lb, ub, step)
        self.ys: np.array = np.arange(lb, ub, step)
        self.fun = fun
        self.cmap = plt.cm.coolwarm  # 'viridis'

    def plot(self, title: str) -> None:
        xs = self.xs
        ys = self.ys

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for x and y
        x_grid, y_grid = np.meshgrid(xs, ys)

        # Calculate z values using the provided function
        # z_grid = np.zeros(x_grid.shape)
        # for i,x in enumerate(xs):
        #     for j,y in enumerate(ys):
        #         z_grid[i,j] = self.fun([x,y])

        # Calculate z values using the provided function (vectorized)
        xy_pairs = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        z_grid = np.array([self.fun(xy)
                          for xy in xy_pairs]).reshape(x_grid.shape)

        ax.plot_surface(x_grid, y_grid, z_grid, cmap=self.cmap)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.title(title)

        plt.show()

    def animate_points(self, title: str, points_to_animate):
        xs = self.xs
        ys = self.ys

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', elev=35, azim=-125)

        x_grid, y_grid = np.meshgrid(xs, ys)
        xy_pairs = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        z_grid = np.array([self.fun(xy)
                          for xy in xy_pairs]).reshape(x_grid.shape)

        plt.title(title)

        def update(frame):
            # Animate the points
            if frame < len(points_to_animate):
                ax.clear()

                # Redraw the surface
                ax.plot_surface(x_grid, y_grid, z_grid,
                                cmap=self.cmap, alpha=0.5, zorder=1)

                # Plot the points up to the current frame, except the first frame
                if frame != 0:
                    history_points = [
                        point for point in points_to_animate[:frame]]
                    x_values, y_values, z_values = zip(*history_points)
                    ax.scatter(x_values, y_values, z_values, s=15,
                               c='black', label='History individuals', zorder=4)

                # Plot current point (except the last frame)
                current_individual_label = 'Current Individual'
                if frame < len(points_to_animate) - 1:
                    current_point = points_to_animate[frame + 1]
                    ax.scatter(*current_point, s=20, c='red',
                               label=current_individual_label, zorder=4)
                else:
                    current_point = points_to_animate[frame]
                    ax.scatter(*current_point, s=20, c='red',
                               label=current_individual_label, zorder=4)

                # Redraw labels and legend
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.legend()

                # Rotate
                azim = ax.azim
                azim = (azim + 1) % 360
                ax.azim = azim

            # Rotate the plot after the animation of the points has finished
            else:
                azim = ax.azim
                azim = (azim + 1) % 360
                ax.azim = azim

        # Animate the points
        anim = FuncAnimation(fig, update, frames=len(
            points_to_animate) + 180, repeat=False, blit=False)

        # plt.show()

        return anim

    def animate_points360(self, title: str, points_to_animate):
        xs = self.xs
        ys = self.ys

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', elev=35, azim=-125)

        x_grid, y_grid = np.meshgrid(xs, ys)
        xy_pairs = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        z_grid = np.array([self.fun(xy)
                          for xy in xy_pairs]).reshape(x_grid.shape)

        plt.title(title)

        def rotate():
            azim = ax.azim
            azim = (azim + 1) % 360
            ax.azim = azim

        step = int(360 / len(points_to_animate))

        def update(frame):
            global cnt
            if frame == 0:
                cnt = 0
            # Animate the points
            if frame % step == 0:

                if cnt > len(points_to_animate) - 1:
                    rotate()
                    return

                ax.clear()

                # Redraw the surface
                ax.plot_surface(x_grid, y_grid, z_grid,
                                cmap=self.cmap, alpha=0.5, zorder=1)

                # Plot the points up to the current frame, except the first frame
                if cnt != 0:
                    history_points = [
                        point for point in points_to_animate[:cnt]]
                    x_values, y_values, z_values = zip(*history_points)
                    ax.scatter(x_values, y_values, z_values, s=15,
                               c='black', label='History individuals', zorder=4)

                # Plot current point (except the last frame)
                current_individual_label = 'Current Individual'
                if cnt < len(points_to_animate) - 1:
                    current_point = points_to_animate[cnt + 1]
                    ax.scatter(*current_point, s=20, c='red',
                               label=current_individual_label, zorder=4)
                else:
                    current_point = points_to_animate[cnt]
                    ax.scatter(*current_point, s=20, c='red',
                               label=current_individual_label, zorder=4)

                # Redraw labels and legend
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.legend()

                cnt += 1

                # Rotate the plot
                rotate()

            # Rotate the plot
            else:
                rotate()

        # Animate the points
        anim = FuncAnimation(fig, update, frames=360, repeat=False, blit=False)

        # plt.show()

        return anim


def plot_my_functions() -> None:
    for fun in F:
        Graph(interval=Function.get_interval(fun),
              fun=Function.get(fun)).plot(str.capitalize(fun.name))


def animate_optimizer(function: F, optimizer: Opt, optimizer_args: list = [], format: Union[None, str] = "mp4"):
    fun = Function.get(function)
    interval = Function.get_interval(function)
    graph = Graph(interval, fun)
    optimizer_algorithm = Optimizer.factory(optimizer, interval, fun)
    points = optimizer_algorithm.run(*optimizer_args)
    anim = graph.animate_points360(str.capitalize(function.name), points)

    if format == "gif":
        anim.save(f'{function.name}_{str.lower(optimizer.name)}.gif',
                  writer='imagemagick',
                  fps=24,
                  progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'))

    if format == "mp4":
        anim.save(f'{function.name}_{str.lower(optimizer.name)}.mp4',
                  fps=15,
                  extra_args=['-vcodec', 'libx264'],
                  progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'))


def plot_sphere(r: float = 1, title: str = "Sphere") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.title(title)

    plt.show()
