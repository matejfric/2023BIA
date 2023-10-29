import numpy as np
from typing import Callable, Union
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from interval import Interval
from function import Function, F
from solution import Optimizer, Opt
from point import Point


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

        self.plot_surface(ax, x_grid, y_grid, z_grid)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.title(title)

        plt.show()

    def plot_surface(self, ax: plt.Axes, x_grid, y_grid, z_grid):
        return ax.plot_surface(x_grid, y_grid, z_grid, linewidth=0.05,edgecolor='k',
                                cmap=self.cmap, alpha=0.7, zorder=1)

    def animate_points_contour(self, title: str, points, batch_size=20):
        xs = self.xs
        ys = self.ys

        fig = plt.figure()
        ax = fig.add_subplot(111)

        x_grid, y_grid = np.meshgrid(xs, ys)
        xy_pairs = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        z_grid = np.array([self.fun(xy)
                          for xy in xy_pairs]).reshape(x_grid.shape)

        plt.title(title)

        cset = ax.contourf(x_grid, y_grid, z_grid, cmap=self.cmap)
        plt.colorbar(cset, ax=ax, shrink=0.5, aspect=5)

        # Create a function to update the plot with a batch of points
        def update(frame):
            ax.clear()
            ax.contourf(x_grid, y_grid, z_grid, cmap=self.cmap)
            batch_points = points[frame * batch_size:(frame + 1) * batch_size + 1]

            if batch_points:
                batch_x, batch_y, batch_z = zip(*batch_points)
                ax.scatter(batch_x, batch_y, s=15, c='black', zorder=2)

                idx_best = np.argmin(batch_z)
                x, y, z = batch_x[idx_best], batch_y[idx_best], batch_z[idx_best]
                ax.scatter(x, y, s=15, c='red', zorder=2)
                ax.text(x, y, s=f"(x={x:.2f}, y={y:.2f}, z={z:.2f})", va='top', ha='left')

            ax.set_xlabel('x')
            ax.set_ylabel('y')

        num_frames = len(points) // batch_size + 1

        # Create the animation
        anim = FuncAnimation(fig, update, frames=num_frames, repeat=False)

        plt.show()

        return anim

    def plot_points_contour(self, title: str, points):
        xs = self.xs
        ys = self.ys

        fig = plt.figure()
        ax = fig.add_subplot(111)

        x_grid, y_grid = np.meshgrid(xs, ys)
        xy_pairs = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        z_grid = np.array([self.fun(xy)
                          for xy in xy_pairs]).reshape(x_grid.shape)

        plt.title(title)

        cmap = plt.get_cmap('coolwarm')

        cset = ax.contourf(x_grid, y_grid, z_grid, cmap=cmap)
        plt.colorbar(cset, ax=ax, shrink=0.5, aspect=5)

        # Plot the search history
        points = [point for point in points]
        x_values, y_values, z_values = zip(*points)
        ax.scatter(x_values, y_values, s=15,
                   c='black', label='History individuals', zorder=2)

        # Plot the solution
        idx_best = np.argmin(z_values)
        x, y, z = x_values[idx_best], y_values[idx_best], z_values[idx_best]
        ax.scatter(x, y, s=30, c='red', label='Solution', zorder=3)
        ax.text(x, y, s=f"(x={x:.2f}, y={y:.2f}, z={z:.2f})", va='top', ha='left')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.legend()
        plt.show()

    def plot_points(self, title: str, points):
        xs = self.xs
        ys = self.ys

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', elev=35, azim=-125)

        x_grid, y_grid = np.meshgrid(xs, ys)
        xy_pairs = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        z_grid = np.array([self.fun(xy)
                          for xy in xy_pairs]).reshape(x_grid.shape)

        plt.title(title)

        surf = self.plot_surface(ax,x_grid, y_grid, z_grid)
        
        # Plot the search history
        points = [point for point in points]
        x_values, y_values, z_values = zip(*points)
        ax.scatter(x_values, y_values, z_values, s=15,
                 c='black', label='History individuals', zorder=np.inf)
        
        # Plot the solution
        idx_best = np.argmin(z_values)
        x, y, z = x_values[idx_best], y_values[idx_best], z_values[idx_best]
        ax.scatter(x,y,z, s=30, c='red', label='Solution', zorder=4)
        ax.text(x,y,z, s=f"(x={x:.2f}, y={y:.2f}, z={z:.2f})", va = 'top', ha = 'left')

        fig.colorbar(surf, ax = ax,shrink = 0.5,aspect = 5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()
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

                self.plot_surface(ax, x_grid, y_grid, z_grid)

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
        if step == 0:
            step = 1

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
                self.plot_surface(ax,x_grid, y_grid, z_grid)

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


def evaluate_points(function: F, optimizer: Opt, optimizer_args: Union[list, dict] = [])-> (Graph, list[Point]):
    fun = Function.get(function)
    interval = Function.get_interval(function)
    graph = Graph(interval, fun)
    optimizer_algorithm = Optimizer.factory(optimizer, interval, fun)

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
    
    return (graph, points)

def plot_optimizer_contour(function: F, optimizer: Opt, optimizer_args: Union[list, dict] = []):
    graph, points = evaluate_points(function, optimizer, optimizer_args)
    graph.plot_points_contour(str.capitalize(function.name), points)

def plot_optimizer(function: F, optimizer: Opt, optimizer_args: Union[list, dict] = []):
    graph, points = evaluate_points(function, optimizer, optimizer_args)
    graph.plot_points(str.capitalize(function.name), points)

def animate_optimizer_contour(function: F, optimizer: Opt, optimizer_args: Union[list, dict] = [], format: str = "mp4"):
    graph: Graph
    points: list[Point]
    graph, points = evaluate_points(function, optimizer, optimizer_args)
    
    if 'swarm_size' in optimizer_args:
        anim = graph.animate_points_contour(str.capitalize(function.name), points, optimizer_args['swarm_size'])
    else:
        anim = graph.animate_points_contour(str.capitalize(function.name), points, 1)

    if format == None:
        format = "mp4"

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

def animate_optimizer(function: F, optimizer: Opt, optimizer_args: Union[list, dict] = [], format: str = "mp4"):
    graph: Graph
    points: list[Point]
    graph, points = evaluate_points(function, optimizer, optimizer_args)
    anim = graph.animate_points360(str.capitalize(function.name), points)

    if format == None:
        format = "mp4"

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
