"""Define a dynamical system for a 13D quadrotor"""
from typing import Tuple, List, Optional

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem

from .utils import grav, Scenario


from tqdm import tqdm
import torch.nn as nn
import math

class Quad13D(ControlAffineSystem):
    """
    Represents a planar quadrotor.

    The system has state

        x = [px, py, pz, vx, vy, vz, phi, theta, psi]

    representing the position, orientation, and velocities of the quadrotor, and it
    has control inputs

        u = [f, WX, WY, WZ]

    The system is parameterized by
        m: mass

    NOTE: Z is defined as positive downwards
    """

    # Number of states and controls
    N_DIMS = 13
    N_CONTROLS = 4

    # State indices
    PX = 0
    PY = 1
    PZ = 2

    QW = 3
    QX = 4
    QY = 5
    QZ = 6

    VX = 7
    VY = 8
    VZ = 9

    WX = 10
    WY = 11
    WZ = 12

    # Control indices
    F = 0
    PHI_ACC = 1
    THETA_ACC = 2
    PSI_ACC = 3

    def __init__(
        self,
        state_limits,
        control_limits,
        safe_mask,
        failure_mask,
        f,
        g,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
    ):
        """
        Initialize the quadrotor.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        self.dynamics_=Quadrotor(0.5, 20.0, 'avoid')
        self.mpc = MPC(horizon=20, receding_horizon=1, dT=0.025, num_samples=100, 
              dynamics_=self.dynamics_, device='cuda', mode="MPC", sample_mode="gaussian",
              style='direct',num_iterative_refinement=2)
        self.mpc.T=0.5
        super().__init__(nominal_params, dt, controller_dt, use_linearized_controller=False)

        self.custom_state_limits = state_limits
        self.custom_control_limits = control_limits
        self.custom_safe_mask = safe_mask
        self.custom_failure_mask = failure_mask
        self.custom_f = f
        self.custom_g = g

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "g" in params

        # Make sure all parameters are physically valid
        valid = valid and params["g"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return Quad13D.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return Quad13D.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        return self.custom_state_limits()

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        return self.custom_control_limits()

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        return self.custom_safe_mask(x).to(device=x.device)

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        return self.custom_failure_mask(x).to(device=x.device)

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = (x[:,:3]-torch.tensor([3, 3, 0], device=x.device)).norm(dim=-1) <= 0.2


        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        return self.custom_f(x).unsqueeze(-1).to(device=x.device)

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        return self.custom_g(x).to(device=x.device)

    @property
    def u_eq(self):
        u_eq = torch.zeros((1, self.n_controls))
        u_eq[0, Quad13D.F] = 9.8
        return u_eq

    @property
    def goal_point(self):
        goal = torch.zeros((1, self.n_dims))
        goal[0, Quad13D.PX]=3
        goal[0, Quad13D.PY]=3
        goal[0, Quad13D.PZ]=0
        goal[0, Quad13D.QW]=1
        return goal
    
    def u_nominal(self, x, params = None):
        state_trajs, lxs, num_iters, best_controls = self.mpc.get_opt_trajs(x, None, 0.5)
        u_nom=best_controls[:,0,:]*1.0
        u_nom[u_nom==np.nan]=0.0
        u_nom=torch.zeros((x.shape[0],self.n_controls))
        return u_nom
    


    
class Quadrotor():
    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR
        self.reach_fn_weight = 1.
        self.avoid_fn_weight = 0.3
        self.state_range_ = torch.tensor([
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            ])
        self.control_range_ =torch.tensor([[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]).cuda()
        self.eps_var=torch.tensor([20,8,8,4]).cuda()
        self.control_init= torch.tensor([-self.Gz*0.0,0,0,0]).cuda()

        if set_mode=='reach_avoid':
            l_type='brat_hjivi'
        else:
            l_type='brt_hjivi'

        self.name= 'Quadrotor' 
        self.loss_type = l_type
        self.set_mode = set_mode
        self.state_dim = 13
        self.input_dim = 14
        self.control_dim = 4
        self.disturbance_dim = 0
        self.state_mean = torch.tensor([0 for i in range(13)])
        self.state_var = torch.tensor([3.0, 3.0, 3.0, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5])
        self.value_mean = (math.sqrt(3.0**2 + 3.0**2) - 2 * self.collisionR) / 2
        self.value_var = math.sqrt(3.0**2 + 3.0**2)/2
        self.value_normto = 0.02
        self.deepReach_model = 'exact'

    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2,dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x
    def set_model(self, deepreach_model):
        self.deepReach_model = deepreach_model
    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)
                          ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord*1.0
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)
                          ) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == 'diff':
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact':
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact_diff':
            # V(x,t)= l(x) + NN(x,t) - NN(x,0)
            # print(input.shape, output.shape)
            output0 = output[0].squeeze(dim=-1)
            output1 = output[1].squeeze(dim=-1)
            return (output0 - output1) * self.value_var / self.value_normto + self.boundary_fn(self.input_to_coord(input[0].detach())[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean


    def quaternion_invert(self,quaternion: torch.Tensor) -> torch.Tensor:
        """
        Given a quaternion representing rotation, get the quaternion representing
        its inverse.

        Args:
            quaternion: Quaternions as tensor of shape (..., 4), with real part
                first, which must be versors (unit quaternions).

        Returns:
            The inverse, a tensor of quaternions of shape (..., 4).
        """

        scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
        return quaternion * scaling


    def quaternion_raw_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply two quaternions.
        Usual torch rules for broadcasting apply.

        Args:
            a: Quaternions as tensor of shape (..., 4), real part first.
            b: Quaternions as tensor of shape (..., 4), real part first.

        Returns:
            The product of a and b, a tensor of quaternions shape (..., 4).
        """
        aw, ax, ay, az = torch.unbind(a, -1)
        bw, bx, by, bz = torch.unbind(b, -1)
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack((ow, ox, oy, oz), -1)


    def quaternion_apply(self,quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """
        Apply the rotation given by a quaternion to a 3D point.
        Usual torch rules for broadcasting apply.

        Args:
            quaternion: Tensor of quaternions, real part first, of shape (..., 4).
            point: Tensor of 3D points of shape (..., 3).

        Returns:
            Tensor of rotated points of shape (..., 3).
        """
        if point.size(-1) != 3:
            raise ValueError(f"Points are not in 3D, {point.shape}.")
        real_parts = point.new_zeros(point.shape[:-1] + (1,))
        point_as_quaternion = torch.cat((real_parts, point), -1)
        out = self.quaternion_raw_multiply(
            self.quaternion_raw_multiply(quaternion, point_as_quaternion),
            self.quaternion_invert(quaternion),
        )
        return out[..., 1:]

    def clamp_control(self, state, control):
        return control
    
    def clamp_state_input(self, state_input):
        return state_input
    
    def clamp_verification_state(self, state):
        return state
    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return [
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]
    
    def state_verification_range(self):
        return [
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def periodic_transform_fn(self, input):
        return input
    
    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]) * 1.0

        return dsdt
    
    def dist_to_cylinder(self, state, a, b):
        '''for cylinder with full body collision'''
        state_=state*1.0
        state_[...,0]=state_[...,0]- a
        state_[...,1]=state_[...,1]- b

        # create normal vector
        v = torch.zeros_like(state_[..., 4:7])
        v[..., 2] = 1
        v = self.quaternion_apply(state_[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state_[..., 0]
        py = state_[..., 1]

        # get full body distance
        dist = torch.norm(state_[..., :2], dim=-1)
        # return dist- self.collisionR
        dist = dist- torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR
    
    def reach_fn(self, state):
        state_=state*1.0
        state_[...,0]=state_[...,0] - 0.
        state_[...,1]=state_[...,1]
        return (torch.norm(state[..., :2], dim=-1)-0.3)*self.reach_fn_weight

    def avoid_fn(self, state):
        return self.avoid_fn_weight*torch.minimum(self.dist_to_cylinder(state,0.0,0.75), self.dist_to_cylinder(state,0.0,-0.75))

    def boundary_fn(self, state):
        if self.set_mode=='avoid':
            # return self.avoid_fn(state)
            return self.dist_to_cylinder(state,0.0,0.0)
            # return torch.minimum(self.dist_to_cylinder(state,0.0,0.75), self.dist_to_cylinder(state,0.0,-0.75))
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    # def sample_target_state(self, num_samples):
    #     raise NotImplementedError
    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-1, 1]
        target_state_range[1] = [-0.25, 0.25]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])
    
    def cost_fn(self, state_traj):
        if self.set_mode=='avoid':
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        else:
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values


    # def cost_fn(self, state_traj):
    #     return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham -= torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham -= torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        else:
            raise NotImplementedError

        return ham

    def optimal_control(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = -self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = -self.dwx_max * torch.sign(dvds[..., 10])
            u3 = -self.dwy_max * torch.sign(dvds[..., 11])
            u4 = -self.dwz_max * torch.sign(dvds[..., 12])
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        return {
            # 'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # 'state_slices': [0, 0, 0, 0.813, -0.022, 0.581, -0.019, 0, 3.5, 0, 2, -1, 1.5],
            # 'state_slices': [0,  0,  1.16,  0.11,      0.72,   -0.40, -0.55,   1.15,   -1.39,  0.06,  -0.99,  3.42,   -2.32],
            'state_slices': [0.96,  1.18,  0.54,  0.44, -0.45,  0.27, -0.73, -2.83, -1.07, -3.34, 3.19, -2.80,  3.43],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }
    


class MPC:
    def __init__(self, dT, horizon, receding_horizon, num_samples, dynamics_, device, mode="MPC",
                  sample_mode="gaussian", lambda_=0.01, style= "direct",num_iterative_refinement=1):
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.receding_horizon = receding_horizon
        self.dynamics_=dynamics_

        self.dT = dT


        
        self.lambda_ = lambda_

        self.mode=mode
        self.sample_mode=sample_mode
        self.style = style # choice: receding, direct
        self.num_iterative_refinement=num_iterative_refinement


    def init_control_tensors(self):
        self.control_init =self.dynamics_.control_init.unsqueeze(0).repeat(self.batch_size,1)
        self.control_tensors = self.control_init.unsqueeze(1).repeat(1,self.horizon,1) # A * H * D_u
    

    def get_next_step_state(self, state, controls):
        current_dsdt = self.dynamics_.dsdt(
            state, controls, None)
        next_states= self.dynamics_.equivalent_wrapped_state(state + current_dsdt*self.dT)
        # next_states = torch.clamp(next_states, self.dynamics_.state_range_[..., 0], self.dynamics_.state_range_[..., 1])
        return next_states

    def rollout_dynamics(self, initial_state_tensor,eps_var_factor=1):
        # returns the state trajectory list and swith collision
        if self.sample_mode == "gaussian":
            epsilon_tensor = torch.randn(
                self.batch_size, self.num_samples, self.horizon, self.dynamics_.control_dim).to(self.device)*torch.sqrt(self.dynamics_.eps_var)*eps_var_factor # A * N * H * D_u

            epsilon_tensor[:, 0, ...] = 0.0  # always include the nominal trajectory
            if self.num_iterative_refinement==0:
                epsilon_tensor*=0.0
            permuted_controls = self.control_tensors.unsqueeze(1).repeat(1, 
                self.num_samples, 1, 1) + epsilon_tensor *1.0 # A * N * H * D_u
        elif self.sample_mode == "binary":
            permuted_controls = torch.sign(torch.empty(self.batch_size,self.num_samples, self.horizon, self.dynamics_.control_dim).uniform_(-1, 1)).to(self.device)
            permuted_controls [:, 0, ...] = self.control_tensors*1.0 # always include the nominal trajectory

        # clamp control
        permuted_controls = torch.clamp(permuted_controls, self.dynamics_.control_range_[..., 0], self.dynamics_.control_range_[..., 1])

        # rollout trajs
        state_trajs = torch.zeros((self.batch_size, self.num_samples, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * N * H * D
        state_trajs[:, :, 0, :] = initial_state_tensor.unsqueeze(1).repeat(1, self.num_samples, 1) # A * N * D
        
        for k in range(self.horizon):
            permuted_controls[:, :, k, :]=self.dynamics_.clamp_control(state_trajs[:, :, k, :], permuted_controls[:, :, k, :])
            state_trajs[:, :, k+1,:]= self.get_next_step_state(
                state_trajs[:, :, k, :], permuted_controls[:, :, k, :])

        return state_trajs, permuted_controls
    

    def rollout_nominal_trajs(self,initial_state_tensor):
        # rollout trajs
        state_trajs = torch.zeros((self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
        state_trajs[:, 0, :] = initial_state_tensor*1.0 # A * D

        for k in range(self.horizon):

            state_trajs[:, k+1,:]= self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k, :])
        return state_trajs
            
    def update_control_tensor(self, state_trajs, permuted_controls, receding=True):   
        costs = self.dynamics_.cost_fn(state_trajs) # A * N
        # if t_remaining>0.0:
        #     traj_times=torch.ones(self.batch_size,self.num_samples,1).to(self.device)*t_remaining
        #     state_trajs_clamped = torch.clamp(state_trajs[:, :, -1, :], torch.tensor(self.dynamics_.state_test_range(
        #                     )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])
        #     # state_trajs_clamped = state_trajs[:, :, -1, :]*1.0
        #     traj_coords = torch.cat(
        #         (traj_times, state_trajs_clamped), dim=-1)
        #     traj_policy_results = policy(
        #         {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
        #     terminal_values=self.dynamics_.io_to_value(traj_policy_results['model_in'].detach(
        #         ), traj_policy_results['model_out'].squeeze(dim=-1).detach())
        #     costs = torch.minimum(costs, terminal_values)

        # costs += self.dynamics_.boundary_fn(state_trajs[:,:,-1,:])*1e-6
        if self.mode=="MPC":
            # just use the best control
            if self.dynamics_.set_mode == 'avoid':
                best_costs, best_idx=costs.max(1)
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                best_costs, best_idx=costs.min(1)
            else:
                raise NotImplementedError
            expanded_idx = best_idx[...,None, None, None].expand(-1, -1, permuted_controls.size(2), permuted_controls.size(3))  

            best_controls = torch.gather(permuted_controls, dim=1, index=expanded_idx).squeeze(1) # A * H * D_u
            self.control_tensors = best_controls*1.0
            expanded_idx_traj = best_idx[...,None, None, None].expand(-1, -1, state_trajs.size(2), state_trajs.size(3))  
            best_traj= torch.gather(state_trajs, dim=1, index=expanded_idx_traj).squeeze(1)
        elif self.mode=="MPPI":
            # use weighted average
            if self.dynamics_.set_mode == 'avoid':
                exp_terms = torch.exp((1/self.lambda_)*costs) # A * N
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                exp_terms = torch.exp((1/self.lambda_)*-costs) # A * N
            else:
                raise NotImplementedError
            
            den = torch.sum(exp_terms, dim=-1) # A

            num = torch.sum(exp_terms[:, :, None, None].repeat(1,1,self.horizon, self.dynamics_.control_dim) * permuted_controls, dim=1) # A * H * D_u


            self.control_tensors = num/den[:,None,None]

            self.control_tensors = torch.clamp(
                self.control_tensors, self.dynamics_.control_range_[..., 0], self.dynamics_.control_range_[..., 1])
        else:
            raise NotImplementedError
        # update controls
        current_controls = self.control_tensors[:, :self.receding_horizon, :]
        if receding:
          self.control_tensors[:, :self.horizon-self.receding_horizon,
                              :] = self.control_tensors[:,self.receding_horizon:, :]
          self.control_tensors[:, self.horizon-self.receding_horizon:, :] = self.control_init.unsqueeze(1).repeat(1,self.receding_horizon,1) # A * H_r * D_u 

        return current_controls, best_traj, best_costs
    

    def rollout_with_policy(self, initial_condition_tensor, policy):
        state_trajs = torch.zeros((self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
        state_trajs_clamped = torch.zeros((self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
        state_trajs[:, 0, :] = initial_condition_tensor*1.0
        state_trajs_clamped[:, 0, :]=state_trajs[:, 0, :]*1.0
        traj_times=torch.ones(self.batch_size,1).to(self.device)*self.horizon*self.dT
        for k in range(self.horizon):
            
            traj_coords = torch.cat(
                (traj_times, state_trajs_clamped[:, k, :]), dim=-1)
            traj_policy_results = policy(
                {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
            traj_dvs = self.dynamics_.io_to_dv(
                traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()
            if k==0:
                values_pred=self.dynamics_.io_to_value(traj_policy_results['model_in'].detach(
                        ), traj_policy_results['model_out'].squeeze(dim=-1).detach())
            self.control_tensors[:, k, :] = self.dynamics_.optimal_control(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device))
    
            state_trajs[:, k+1,:] = self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k, :])

            state_trajs_clamped[:, k+1,:] = torch.clamp(state_trajs[:, k+1,:], torch.tensor(self.dynamics_.state_test_range(
                    )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])
            traj_times=traj_times-self.dT

        # nominal_traj=self.rollout_nominal_trajs(initial_condition_tensor).cpu()
        state_trajs, permuted_controls = self.rollout_dynamics(initial_condition_tensor,1.0)

        current_controls, best_traj, best_costs = self.update_control_tensor(
            state_trajs, permuted_controls, receding=False) 
        
        
        # we want to have more critical trajectories!
        # unsafe_cost_safe_value_indeces = torch.argwhere(
        #                 torch.logical_and(best_costs < 0, values_pred >= 0)).detach().squeeze(-1)

        # idxs = torch.randperm(best_traj.shape[0])[:int(best_traj.shape[0]/3)]
        # # print(best_costs.shape, values_pred.shape, best_traj.shape, best_traj[idxs, ...].shape,best_traj[unsafe_cost_safe_value_indeces].shape,unsafe_cost_safe_value_indeces.shape)
        # best_traj=torch.cat([best_traj[idxs, ...], best_traj[unsafe_cost_safe_value_indeces]],dim=0)
        return best_traj
    
    def get_control(self, initial_condition_tensor, num_iterative_refinement=1, policy=None, t_remaining=None):
        
        if self.style == 'direct':
            # last_best_costs=torch.ones(self.batch_size).to(self.device)*torch.finfo().max
            if num_iterative_refinement==-1: # rollout using the policy
                best_traj = self.rollout_with_policy(initial_condition_tensor,policy)
            for i in range(num_iterative_refinement+1):
                if i==1 and policy is not None:
                    # Rollout with the policy to get initial control guess. 
                    # Use the opt control from the first iteration for steps before the refinement horizon
                    state_trajs = torch.zeros((self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
                    state_trajs_clamped = torch.zeros((self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
                    state_trajs[:, 0, :] = initial_condition_tensor*1.0
                    state_trajs_clamped[:, 0, :]=state_trajs[:, 0, :]*1.0
                    traj_times=torch.ones(self.batch_size,1).to(self.device)*self.horizon*self.dT
                    for k in range(self.horizon):
                        if k>=self.num_refinement_horizon:
                            traj_coords = torch.cat(
                                (traj_times, state_trajs_clamped[:, k, :]), dim=-1)
                            traj_policy_results = policy(
                                {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
                            traj_dvs = self.dynamics_.io_to_dv(
                                traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()
                            self.control_tensors[:, k, :] = self.dynamics_.optimal_control(
                                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device))
                
                        state_trajs[:, k+1,:] = self.get_next_step_state(
                            state_trajs[:, k, :], self.control_tensors[:, k, :])
                        # TODO: Decide whether we should clamp the states here
                        state_trajs_clamped[:, k+1,:] = torch.clamp(state_trajs[:, k+1,:], torch.tensor(self.dynamics_.state_test_range(
                                )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])
                        traj_times=traj_times-self.dT
                # eps_var_factor=0.95**i
                eps_var_factor=1
                # nominal_traj=self.rollout_nominal_trajs(initial_condition_tensor).cpu()
                state_trajs, permuted_controls = self.rollout_dynamics(initial_condition_tensor,eps_var_factor)
                self.all_state_trajs=state_trajs.detach().cpu()*1.0
                current_controls, best_traj, best_costs = self.update_control_tensor(
                    state_trajs, permuted_controls, receding=False) 
                
                
            return self.control_tensors, best_traj
        elif self.style == 'receding':
            # initial_condition_tensor: A*D
            state_trajs, permuted_controls = self.rollout_dynamics(initial_condition_tensor)

            current_controls, best_traj = self.update_control_tensor(
                state_trajs, permuted_controls) 
        
            return current_controls, best_traj

    def get_opt_trajs(self,initial_condition_tensor, policy=None, t=0.0):
        self.batch_size=initial_condition_tensor.shape[0]
        num_iters = math.ceil((self.T)/self.dT)
        self.horizon = math.ceil((self.T)/self.dT)
        self.num_refinement_horizon =  math.ceil((self.T-t)/self.dT)
        if self.style == 'direct':
            
            self.init_control_tensors()
            best_controls, best_trajs = self.get_control(
                    initial_condition_tensor, self.num_iterative_refinement, policy, t_remaining=t)
            
            if self.dynamics_.set_mode in ['avoid', 'reach']:
                lxs = self.dynamics_.boundary_fn(best_trajs)   
                return best_trajs, lxs, num_iters, best_controls
            elif self.dynamics_.set_mode == 'reach_avoid':
                avoid_values=self.dynamics_.avoid_fn(best_trajs) 
                reach_values=self.dynamics_.reach_fn(best_trajs) 
                return best_trajs, avoid_values, reach_values, num_iters, best_controls
            else:
                raise NotImplementedError
            

        elif self.style == 'receding':
            if self.dynamics_.set_mode =='reach_avoid':
                raise NotImplementedError

            state_trajs = torch.zeros(( self.batch_size, num_iters+1, self.dynamics_.state_dim)).to(self.device)  # A*H*D
            state_trajs[:, 0, :] = initial_condition_tensor

            self.init_control_tensors()

            lxs=torch.zeros(self.batch_size, num_iters+1).to(self.device)

            for i in range(int(num_iters/self.receding_horizon)):
                best_controls,_ = self.get_control(
                        state_trajs[:,i, :])
                for k in range(self.receding_horizon):
                    lxs[:,i*self.receding_horizon+k] = self.dynamics_.boundary_fn(
                                            state_trajs[:, i*self.receding_horizon+k, :]) 
                    state_trajs[:,i*self.receding_horizon+1+k,:] = self.get_next_step_state(
                        state_trajs[:,i*self.receding_horizon+k,:], best_controls[:, k, :])
            lxs[:,-1] = self.dynamics_.boundary_fn(state_trajs[:, -1, :]) 
            return state_trajs, lxs, num_iters
        else:
            return NotImplementedError
        
    
    def get_batch_data(self, initial_condition_tensor, T, policy=None, t=0.0, style="random"):
        self.T=T*1.0
        self.batch_size=initial_condition_tensor.shape[0]
        if self.dynamics_.set_mode in ['avoid', 'reach']:
            state_trajs, lxs, num_iters, best_controls = self.get_opt_trajs(initial_condition_tensor, policy, t)
            costs,_=torch.min(lxs,dim=-1)
            
        elif self.dynamics_.set_mode == 'reach_avoid':
            state_trajs, avoid_values, reach_values, num_iters, best_controls = self.get_opt_trajs(initial_condition_tensor,policy, t)
            costs=torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values
        else:
            raise NotImplementedError
 

        # generating MPC dataset: {..., (t, x, J, u), ...}
        # if style=='terminal': # only generate terminal time samples
        #     num_iters=1
        coords=torch.zeros(self.batch_size* num_iters, self.dynamics_.state_dim+1).to(self.device)
        value_labels=torch.zeros(self.batch_size* num_iters).to(self.device)
        control_labels=torch.zeros(self.batch_size* num_iters, self.dynamics_.control_dim).to(self.device)
        for i in range(num_iters):
            coords[i*self.batch_size: (i+1)*self.batch_size ,0] = self.T - i * self.dT
            coords[i*self.batch_size: (i+1)*self.batch_size,1:] = state_trajs[:, i, :]
            control_labels[i*self.batch_size: (i+1)*self.batch_size] = best_controls[:,i,:]
            if self.dynamics_.set_mode in ['avoid', 'reach']:
                value_labels[i*self.batch_size: (i+1)*self.batch_size],_ =  torch.min(lxs[..., i:],dim=-1) 
            elif self.dynamics_.set_mode == 'reach_avoid':
                value_labels[i*self.batch_size: (i+1)*self.batch_size] =  \
                            torch.min(torch.maximum(reach_values[..., i:], torch.cummax(-avoid_values[..., i:], dim=-1).values), dim=-1).values
            else:
                raise NotImplementedError
        
            
        
        ##################### only use in range labels ###################################################
        output1 = torch.all(coords[...,1:] >= self.dynamics_.state_range_[
                            :, 0]-0.01, -1, keepdim=False)
        output2 = torch.all(coords[...,1:] <= self.dynamics_.state_range_[
                            :, 1]+0.01, -1, keepdim=False)
        in_range_index = torch.logical_and(torch.logical_and(output1, output2), ~torch.isnan(value_labels))


        coords=coords[in_range_index]
        value_labels=value_labels[in_range_index]
        control_labels=control_labels[in_range_index]
        ###################################################################################################
        coords=self.dynamics_.coord_to_input(coords)

        return costs, state_trajs, coords.detach().cpu().clone(), value_labels.detach().cpu().clone(), control_labels.detach().cpu().clone()
            

