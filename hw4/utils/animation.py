"""
Animations for various dynamical systems using `matplotlib`.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation


def animate(t, x, θ, d=None, extra=None, skip_frames=10):
    """Animate the cart-pole system from given position data.

    All arguments are assumed to be 1-D NumPy arrays, where `x` and `θ` are the
    degrees of freedom of the cart-pole over time `t`.

    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate
        fig, ani = animate(t, x, θ)
        ani.save('cartpole.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    cart_width = 2.0
    cart_height = 1.0
    wheel_radius = 0.3
    wheel_sep = 1.0
    pole_length = 5.0
    mass_radius = 0.25

    # Extra arguments
    AX_IDX = 0
    Y_DATA = 1
    STYLES = 2

    # Figure and axis
    num_rows = 1
    if d is not None:
        num_rows += 1
    if extra is not None:
        for name, item in extra.items():
            if item[AX_IDX] + 1 > num_rows:
                num_rows = item[AX_IDX] + 1
    fig, axes = plt.subplots(num_rows, 1, dpi=100)
    ax = axes[0]
    if d is not None:
        ax_d = axes[1]
    axes_configured = [False for _ in range(num_rows)]
    x_min, x_max = np.min(x) - 1.1 * pole_length, np.max(x) + 1.1 * pole_length
    y_min = -pole_length
    y_max = 1.1 * (wheel_radius + cart_height + pole_length)
    ax.plot([x_min, x_max], [0.0, 0.0], "-", linewidth=1, color="k")[0]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    ax.set_xlabel("x (m)")
    axes_configured[0] = True
    if d is not None:
        ax_d.set_xlim([0, t[-1]])
        ax_d.set_ylim([-6, 6])
        ax_d.set_xlabel("t (s)")
        ax_d.set_ylabel("d (rad/s$^2$)")
        ax_d.axhline(0.0, linewidth=1, color="k", linestyle="--")
        dst = ax_d.plot([], [], linewidth=1, color="k", label="d (rad/s$^2$)")[0]
        axes_configured[1] = True
    if extra is not None:
        for name, item in extra.items():
            ax_extra = axes[item[AX_IDX]]
            if not axes_configured[item[AX_IDX]]:
                ax_extra.set_xlim([0, t[-1]])
                ax_extra.set_ylim([min(item[Y_DATA]), max(item[Y_DATA])])
                ax_extra.set_xlabel("t (s)")
                ax_extra.set_ylabel(name)
                axes_configured[item[AX_IDX]] = True
                item[AX_IDX] = ax_extra.plot([], [], label=name, **item[STYLES])[0]
            else:
                ylim = ax_extra.get_ylim()
                ax_extra.set_ylim([
                    min(ylim[0], min(item[Y_DATA])),
                    max(ylim[1], max(item[Y_DATA]))
                ])
                ylabel = ax_extra.get_ylabel()
                ax_extra.set_ylabel(
                    f"{ylabel}\n{name}"
                )
                item[AX_IDX] = ax_extra.plot([], [], label=name, **item[STYLES])[0]
    for axs in axes:
        ymin, ymax = axs.get_ylim()
        margin = 0.05 * (ymax - ymin)
        axs.set_ylim([ymin - margin, ymax + margin])
        _, labels = axs.get_legend_handles_labels()
        if labels:
            axs.legend()
    fig.tight_layout()

    # Artists
    cart = mpatches.FancyBboxPatch(
        (0.0, 0.0),
        cart_width,
        cart_height,
        facecolor="tab:blue",
        edgecolor="k",
        boxstyle="Round,pad=0.,rounding_size=0.05",
    )
    wheel_left = mpatches.Circle((0.0, 0.0), wheel_radius, color="k")
    wheel_right = mpatches.Circle((0.0, 0.0), wheel_radius, color="k")
    mass = mpatches.Circle((0.0, 0.0), mass_radius, color="k")
    pole = ax.plot([], [], "-", linewidth=3, color="k")[0]
    trace = ax.plot([], [], "--", linewidth=2, color="tab:orange")[0]
    timestamp = ax.text(0.1, 0.9, "", transform=ax.transAxes)

    ax.add_patch(cart)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)
    ax.add_patch(mass)

    def _animate(k, t, x, θ):
        # Geometry
        cart_corner = np.array([x[k] - cart_width / 2, wheel_radius])
        wheel_left_center = np.array([x[k] - wheel_sep / 2, wheel_radius])
        wheel_right_center = np.array([x[k] + wheel_sep / 2, wheel_radius])
        pole_start = np.array([x[k], wheel_radius + cart_height])
        pole_end = pole_start + pole_length * np.array([np.sin(θ[k]), -np.cos(θ[k])])

        # Cart
        cart.set_x(cart_corner[0])
        cart.set_y(cart_corner[1])

        # Wheels
        wheel_left.set_center(wheel_left_center)
        wheel_right.set_center(wheel_right_center)

        # Pendulum
        pole.set_data([pole_start[0], pole_end[0]], [pole_start[1], pole_end[1]])
        mass.set_center(pole_end)
        mass_x = x[: k + 1] + pole_length * np.sin(θ[: k + 1])
        mass_y = wheel_radius + cart_height - pole_length * np.cos(θ[: k + 1])
        trace.set_data(mass_x, mass_y)

        # Time-stamp
        timestamp.set_text("t = {:.1f} s".format(t[k]))

        artists = (cart, wheel_left, wheel_right, pole, mass, trace, timestamp)

        if d is not None:
            dst.set_data(t[:k+1], d[:k+1])
            artists += (dst,)

        if extra is not None:
            for _, item in extra.items():
                item[AX_IDX].set_data(t[:k+1], item[Y_DATA][:k+1])
                artists += (item[AX_IDX],)

        return artists

    dt = t[1] - t[0]
    ani = animation.FuncAnimation(
        fig, _animate, range(0, t.size, skip_frames), fargs=(t, x, θ), interval=dt * 1000 * skip_frames, blit=True
    )
    return fig, ani