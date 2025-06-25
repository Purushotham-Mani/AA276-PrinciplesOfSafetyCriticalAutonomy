import os
import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt

# Define problem ingredients (exercise parts (a), (b), (c)).


class PlanarQuadrotor:

    def __init__(self):
        # Dynamics constants
        # yapf: disable
        self.g = 9.807         # gravity (m / s**2)
        self.m = 2.5           # mass (kg)
        self.l = 1.0           # half-length (m)
        self.Iyy = 1.0         # moment of inertia about the out-of-plane axis (kg * m**2)
        self.Cd_v = 0.25       # translational drag coefficient
        self.Cd_phi = 0.02255  # rotational drag coefficient
        # yapf: enable

        # Control constraints
        self.max_thrust_per_prop = (
            0.75 * self.m * self.g
        )  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = (
            0  # at least until variable-pitch quadrotors become mainstream :D
        )

    def full_dynamics(self, full_state, control):
        """Continuous-time dynamics of a planar quadrotor expressed as an ODE."""
        x, v_x, y, v_y, phi, omega = full_state
        T_1, T_2 = control
        return jnp.array(
            [
                v_x,
                (-(T_1 + T_2) * jnp.sin(phi) - self.Cd_v * v_x) / self.m,
                v_y,
                ((T_1 + T_2) * jnp.cos(phi) - self.Cd_v * v_y) / self.m - self.g,
                omega,
                ((T_2 - T_1) * self.l - self.Cd_phi * omega) / self.Iyy,
            ]
        )

    def dynamics(self, state, control):
        """Reduced (for the purpose of reachable set computation) continuous-time dynamics of a planar quadrotor."""
        y, v_y, phi, omega = state
        T_1, T_2 = control
        return jnp.array(
            [
                v_y,
                ((T_1 + T_2) * jnp.cos(phi) - self.Cd_v * v_y) / self.m - self.g,
                omega,
                ((T_2 - T_1) * self.l - self.Cd_phi * omega) / self.Iyy,
            ]
        )

    def optimal_control(self, state, grad_value):
        """Computes the optimal control realized by the HJ PDE Hamiltonian.

        Args:
            state: An unbatched (!) state vector, an array of shape `(4,)` containing `[y, v_y, phi, omega]`.
            grad_value: An array of shape `(4,)` containing the gradient of the value function at `state`.

        Returns:
            A vector of optimal controls, an array of shape `(2,)` containing `[T_1, T_2]`, that minimizes
            `grad_value @ self.dynamics(state, control)`.
        """
        # PART (a): WRITE YOUR CODE BELOW ###############################################
        # You may find `jnp.where` to be useful; see corresponding numpy docstring:
        # https://numpy.org/doc/stable/reference/generated/numpy.where.html
        y, v_y, phi, omega = state
        g = jnp.array([[0, 0], [jnp.cos(phi)/self.m, jnp.cos(phi)/self.m], [0, 0], [-self.l/self.Iyy, self.l/self.Iyy]])
        g = grad_value @ g
        u = jnp.where(g>0, self.min_thrust_per_prop, self.max_thrust_per_prop)
        return  u
        #################################################################################

    def hamiltonian(self, state, time, value, grad_value):
        """Evaluates the HJ PDE Hamiltonian."""
        del time, value  # unused
        control = self.optimal_control(state, grad_value)
        return grad_value @ self.dynamics(state, control)

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del time, value, grad_value_box  # unused
        y, v_y, phi, omega = state
        return jnp.array(
            [
                jnp.abs(v_y),
                (
                    2 * self.max_thrust_per_prop * jnp.abs(jnp.cos(phi))
                    + self.Cd_v * jnp.abs(v_y)
                )
                / self.m
                + self.g,
                jnp.abs(omega),
                (
                    (self.max_thrust_per_prop - self.min_thrust_per_prop) * self.l
                    + self.Cd_phi * jnp.abs(omega)
                )
                / self.Iyy,
            ]
        )


def target_set(state):
    """A real-valued function such that the zero-sublevel set is the target set.

    Args:
        state: An unbatched (!) state vector, an array of shape `(4,)` containing `[y, v_y, phi, omega]`.

    Returns:
        A scalar, nonpositive iff the state is in the target set.
    """
    # PART (b): WRITE YOUR CODE BELOW ###############################################
    x, phi, xdot, omega = state

    x_ = abs(x - 100) - 20
    phi_ = abs(phi-np.pi) - np.pi/2

    return jnp.maximum(x_, phi_)
    #################################################################################


def envelope_set(state):
    """A real-valued function such that the zero-sublevel set is the operational envelope.

    Args:
        state: An unbatched (!) state vector, an array of shape `(4,)` containing `[y, v_y, phi, omega]`.

    Returns:
        A scalar, nonpositive iff the state is in the operational envelope.
    """
    # PART (c): WRITE YOUR CODE BELOW ###############################################
    x, phi, xdot, omega = state
    x_ = abs(x - 60) - 65
    phi_ = abs(phi-np.pi) - np.pi/2

    return jnp.maximum(x_, phi_)
    #################################################################################

cartpole = CartPole()

state_domain = hj.sets.Box(
    lo=np.array([-10.0, 0.0, -10.0, -20.0]),
    hi=np.array([10.0, 2 * np.pi, 10.0, 20.0])
)
grid_resolution = (50, 50, 50, 50)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    state_domain, grid_resolution, periodic_dims=1  # theta is periodic
)

target_values = hj.utils.multivmap(target_set, np.arange(4))(grid.states)
envelope_values = hj.utils.multivmap(envelope_set, np.arange(4))(grid.states)
terminal_values = np.maximum(target_values, envelope_values)

solver_settings = hj.SolverSettings.with_accuracy(
    "medium",  # can/should be changed to "very_high" if running on GPU, or if extra patient
    hamiltonian_postprocessor=lambda x: jnp.minimum(x, 0),
    value_postprocessor=lambda t, x: jnp.maximum(x, envelope_values),
)

# Propagate the HJ PDE _backwards_ in time.
initial_time = 0.0
final_time = -5.0
yn = None
if os.path.exists("hj_reachability_values.npz"):
    yn = (
        input(
            "Existing hj_reachability_values.npz file found from a previous solve; use it (Y/n)? "
        )
        .lower()
        .strip()
    )
if yn is None or yn == "n":
    print("Computing the value function by solving the HJ PDE.")
    values = hj.step(
        solver_settings,
        planar_quadrotor,
        grid,
        initial_time,
        terminal_values,
        final_time,
    ).block_until_ready()
    print("Saving the value function to hj_reachability_values.npz.")
    np.savez("hj_reachability_values.npz", values=values)
else:
    print("Loading previously computed value function from hj_reachability_values.npz.")
    values = np.load("hj_reachability_values.npz")["values"]
grad_values = grid.grad_values(values)

# Utilities for rolling out the optimal controls and visualizing.


@jax.jit
def optimal_step(full_state, dt):
    state = full_state[2:]
    grad_value = grid.interpolate(grad_values, state)
    control = planar_quadrotor.optimal_control(state, grad_value)
    return full_state + dt * planar_quadrotor.full_dynamics(full_state, control)


def optimal_trajectory(full_state, dt=1 / 100, T=5):
    full_states = [full_state]
    t = np.arange(T / dt) * dt
    for _ in t:
        full_states.append(optimal_step(full_states[-1], dt))
    return t, np.array(full_states)