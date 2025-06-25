import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
from hj_reachability import dynamics
from hj_reachability import sets
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interpn
import os
class CartPole(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, mp=1, mc=1, l=1, g=2, u_bar=10):
        self.mp = mp
        self.mc = mc
        self.l = l
        self.g = g

        control_mode = 'max'
        disturbance_mode = 'min'
        control_space = sets.Box(jnp.array([-u_bar]), jnp.array([u_bar]))
        disturbance_space = sets.Box(jnp.array([-5.0]), jnp.array([5.0]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        theta, dtheta = state

        numerator = -( (self.mc + self.mp) * self.g * jnp.sin(theta)
                       + self.mp * self.l * jnp.sin(theta) * jnp.cos(theta) * dtheta**2 )
        denominator = self.l * (self.mc + self.mp * jnp.sin(theta)**2)
        ddtheta = numerator / denominator

        return jnp.array([dtheta, ddtheta])

    def control_jacobian(self, state, time):
        theta, _ = state
        denom = (self.mc + self.mp * jnp.sin(theta)**2) * self.l
        return jnp.array([[0.], [-jnp.cos(theta) / denom]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.], [1.0]])

class ValueFunction:
    def __init__(self):
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
                            hj.sets.Box(np.array([0, -10.]),
                            np.array([2*np.pi, +10.])),
                            (101, 101))

        # Define the implicit function l(x) for the failure set
        failure_values = (np.pi/2) - jnp.abs(self.grid.states[..., 0] - np.pi)
        # Solver settings
        times = np.linspace(0, -10, 201, endpoint=True)
        self.solver_settings = hj.SolverSettings.with_accuracy('very_high',
                                                        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
        
        self.dynamic = CartPole()
        yn = None
        if os.path.exists("hj_reachability_values_2_2.npz"):
            yn = (
                input(
                    "Existing hj_reachability_values_2_2.npz file found from a previous solve; use it (Y/n)? "
                )
                .lower()
                .strip()
            )
        if yn is None or yn == "n":
            print("Computing the value function by solving the HJ PDE.")
            self.values = hj.solve(self.solver_settings, self.dynamic, self.grid, times, failure_values)
            print("Saving the value function to hj_reachability_values_2_2.npz.")
            np.savez("hj_reachability_values_2_2.npz", values=self.values)
        else:
            print("Loading previously computed value function from hj_reachability_values_2_2.npz.")
            self.values = np.load("hj_reachability_values_2_2.npz")["values"]
    
class Controller:
    """
    Controller for the cart-pole system.
    This is a template for you to implement however you desire.
    
    reset(.) is called before each cart-pole simulation.
    u_fn(.) is called at each simulation step.
    data_to_visualize(.) is called after each simulation.

    We provide example code for a random controller.
    """

    def __init__(self):
        self.reset()
        self.V = ValueFunction()
        self.grid = self.V.grid
        self.values = self.V.values
        self.values_converged = self.values[-1]
        self.grads = self.grid.grad_values(self.values_converged, self.V.solver_settings.upwind_scheme)

    def v(self, state):
        return interpn(
            ([np.array(v) for v in self.grid.coordinate_vectors]),
            np.array(self.values_converged),
            state,
            method='linear',
            bounds_error=False,
            fill_value=None
            )
    
    def grad_v(self, state):
        grad_value_x1 = interpn(
            ([np.array(v) for v in self.grid.coordinate_vectors]),
            np.array(self.grads[:, :, 0]),
            state,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        grad_value_x2 = interpn(
            ([np.array(v) for v in self.grid.coordinate_vectors]),
            np.array(self.grads[:, :, 1]),
            state,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        return np.array([grad_value_x1, grad_value_x2])[:,0]
    
    def nominal_control(self, state):
        d = self.d_estimate_history[-1] if len(self.d_estimate_history) > 0 else 0.0
        if d > 1:
            return np.random.uniform(-10, 10)    
        if state[0]>np.pi/2 and state[0]<np.pi-2.3e-1:
            return np.random.uniform(-5, -4)
        if state[0]>=np.pi-2.3e-1 and state[0]<3*np.pi/2:
            return -1
        # else:
    
    def optimal_safety_controller(self, state):
        beta2 =  self.grad_v(state)[1]
        statebeta = self.V.dynamic.control_jacobian(state, 0)[1,0]
        return 10*np.sign(beta2*statebeta).item()
    

    def LR_filter(self, state):
        V = self.v(state)
        if V>1e-1:
            return self.nominal_control(state)
        else:
            return self.optimal_safety_controller(state) 


    def reset(self):
        self.s_history = []
        self.t_history = []
        self.u_history = []
        self.d_estimate_history = []

    def d_est(self, u):
        s = np.array([self.s_history[-1][1] % (2*np.pi), self.s_history[-1][3]])
        sprev = np.array([self.s_history[-2][1] % (2*np.pi), self.s_history[-2][3]])
        
        f = self.V.dynamic.open_loop_dynamics(sprev, 0)[1]
        g = self.V.dynamic.control_jacobian(sprev, 0)[1,0]
        ddtheta = (s[1] - sprev[1]) / (self.t_history[-1] - self.t_history[-2])
        d_estimate = ddtheta - (f + g * u) 
        d_estimate = 0.95*self.d_estimate_history[-1] + 0.05*d_estimate
        
        return d_estimate

    def u_fn(self, s, t):
        """Control function for the cart-pole system.

        Args:
            s (np.ndarray): The current state: [x, theta, x_dot, theta_dot]
                NOTE: you might want to first wrap theta in [0, 2pi]
            t (float): The current time

        Returns:
            u (np.ndarray): The control input [u]
        """
        self.s_history.append(s)
        self.t_history.append(t)
        s = np.array([s[1] % (2*np.pi), s[3]])
        u = self.LR_filter(s)
        self.u_history.append(u)
        if len(self.s_history) < 2:
            d_estimate = 0.0
        else:
            d_estimate = self.d_est(u)
        self.d_estimate_history.append(d_estimate)      
        return np.array([u])            

    def data_to_visualize(self):
        """
        Use this to add any number of data visualizations to the animation.
        This is purely to help you debug, in case you find it helpful.
        See example code below to plot the control on a new axes at axes index 2
        and the disturbance estimate on an existing axes at axes index 1.

        Returns:
            data_to_visualize (dict): Each dictionary entry should have the form:
                'y-axes label' (str): [axes index (int), data to visualize (np.ndarray), line styles (dict)]
        """
        s_history = np.array(self.s_history)
        t_history = np.array(self.t_history)
        u_history = np.array(self.u_history)
        d_estimate_history = np.array(self.d_estimate_history)
        return {
            'u (N)': [2, u_history, {'color': 'k'}],
            '$\\hat{d}$ (rad/s$^2$)': [1, d_estimate_history, {'color': 'k', 'linestyle': '--'}],
            '$\\theta$ (rad)': [3, s_history[:, 1] % (2*np.pi), {'color': 'k'}],
            '$\\theta_\\text{min}$ (rad)': [3, (np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}],
            '$\\theta_\\text{max}$ (rad)': [3, (3*np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}]
        }