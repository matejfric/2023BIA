import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from interval import Interval


class Graph:
    def __init__(self, interval: Interval, fun: Callable):
        lb = interval.lb
        ub = interval.ub
        step = interval.step
        self.xs: np.array = np.arange(lb, ub, step)
        self.ys: np.array = np.arange(lb, ub, step)
        self.fun = fun
        self.cmap = 'viridis'

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
        ax = fig.add_subplot(111, projection='3d')

        x_grid, y_grid = np.meshgrid(xs, ys)
        xy_pairs = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        z_grid = np.array([self.fun(xy)
                          for xy in xy_pairs]).reshape(x_grid.shape)

        plt.title(title)

        def update(frame):
            ax.clear()

            # Redraw the surface
            ax.plot_surface(x_grid, y_grid, z_grid, cmap=self.cmap, alpha=0.7)

            # Plot the points up to the current frame, except the first frame
            if frame != 0:
              history_points = [point for point in points_to_animate[:frame]]
              x_values, y_values, z_values = zip(*history_points)
              ax.scatter(x_values, y_values, z_values, s=30,
                         c='black', label='Blind Search History')

            # Plot current point (except the last frame)
            if frame != len(points_to_animate) - 1:
              current_point = points_to_animate[frame + 1]
              x, y, z = current_point.x, current_point.y, current_point.z
              ax.scatter(x, y, z, s=50, c='red', label='Blind Search Current')

            # Redraw labels and legend
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()

            rotation_step = 1
            ax.view_init(elev=35., azim=(25 + frame * rotation_step) % 360)

        # Animate the points
        anim = FuncAnimation(fig, update, frames=len(
            points_to_animate), repeat=False, blit=False)

        return anim


def plot_sphere(r: float = 1, title: str = "Sphere") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = r

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
