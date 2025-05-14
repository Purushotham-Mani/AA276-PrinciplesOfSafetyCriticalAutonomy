import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

"""
Saves a GIF of your computed value function.
Please read the docstring below for what inputs you need to supply.
"""
def save_values_gif(values, grid, times, save_path='outputs/values.gif'):
    """
    args:
        values: ndarray with shape [
                len(times),
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1])
            ]
        grid: hj.Grid
        times: ndarray with shape [len(times)]
    """
    vbar = 3
    fig, ax = plt.subplots()
    ax.set_title(f'$V(x, {times[0]:3.2f})$')
    ax.set_xlabel('$\\theta$ (rad)')
    ax.set_ylabel('$\\dot{\\theta}$ (rad/s)')
    value_plot = ax.pcolormesh(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[0].T,
        cmap='RdBu',
        vmin=-vbar,
        vmax=+vbar
    )
    plt.colorbar(value_plot, ax=ax)
    global value_contour
    value_contour = ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[0].T,
        levels=0,
        colors='k'
    )

    def update(i):
        ax.set_title(f'$V(x, {times[i]:3.2f})$')
        value_plot.set_array(values[i].T)
        global value_contour
        value_contour.remove()
        value_contour = ax.contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            values[i].T,
            levels=0,
            colors='k'
        )
        return ax
    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=np.arange(len(times)),
        interval=int(1000*(-times[1]))
    )
    with tqdm(total=len(times)) as anim_pbar:
        anim.save(filename=save_path, writer='pillow', progress_callback=lambda i, n: anim_pbar.update(1))
    print(f'SAVED GIF TO: {save_path}')
    plt.close()


"""
Visualizes the value function and safe set boundary.
Please read the docstring below for what inputs you need to supply.
"""
def plot_value_and_safe_set_boundary(values_converged, grid, ax):
    """
    args:
        values_converged: ndarray with shape [
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1])
            ]
        grid: hj.Grid,
        ax: matplotlib axes to plot on
    """
    values_converged_interpolator = RegularGridInterpolator(
        ([np.array(v) for v in grid.coordinate_vectors]),
        np.array(values_converged),
        bounds_error=False,
        fill_value=None
    )
    vbar=3
    vis_thetas = np.linspace(-0.5, +0.5, num=101, endpoint=True)
    vis_theta_dots = np.linspace(-1, +1, num=101, endpoint=True)
    vis_xs = np.stack((np.meshgrid(vis_thetas, vis_theta_dots, indexing='ij')), axis=2)
    vis_values_converged = values_converged_interpolator(vis_xs)
    ax.pcolormesh(
        vis_thetas,
        vis_theta_dots,
        vis_values_converged.T,
        cmap='RdBu',
        vmin=-vbar,
        vmax=vbar
    )
    ax.contour(
        vis_thetas,
        vis_theta_dots,
        vis_values_converged.T,
        levels=[0],
        colors='k'
    )