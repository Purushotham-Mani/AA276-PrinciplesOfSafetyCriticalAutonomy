import torch
from typing import Dict,List,Type,Union

def double_integrator_dynamics(x, u):
    """
    Returns the dynamics (xdot) for a 3-dimensional double integrator system.
    Parameters:
    x (torch.Tensor): State vector [x, y, z, vx, vy, vz]
    u (torch.Tensor): Input vector [ux, uy, uz]

    Returns:
    torch.Tensor: The derivative of the state vector [vx, vy, vz, ax, ay, az]
    """
    assert x.shape == (6,), "State vector x must be of shape (6,)"
    assert u.shape == (3,), "Input vector u must be of shape (3,)"

    # The state vector x consists of position (x, y, z) and velocity (vx, vy, vz)
    pos = x[:3]
    vel = x[3:]

    # The input vector u consists of accelerations (ax, ay, az)
    acc = u

    # The derivative of the state vector is the velocity and acceleration
    xdot = torch.cat((vel, acc))

    return xdot

class SingleIntegrator():
    def __init__(self, device, ndim=3):
        self.ndim = ndim
        self.device = device
        self.rel_deg = 1

    def system(self, x, u=None):
        # Defines the f function
        f = torch.zeros(self.ndim).to(self.device)
        g = torch.eye(self.ndim).to(self.device)
        return f, g

class DoubleIntegrator():
    def __init__(self, drn_prms:Dict[str,Union[int,float,List[float]]], device, ndim=3):
        m = drn_prms["mass"]     
        fn = drn_prms["force_normalized"]
        n_rtr = drn_prms["number_of_rotors"]
        self.tn = fn*n_rtr
        self.m = m
    
        self.ndim = ndim
        self.device = device
        self.rel_deg = 2

    def system(self, x, u=None):
        """
        Defines the f and g functions for the double integrator system.
        x: state vector (position and velocity)
        u: control input (acceleration)
        """
        # Split state vector x into position and velocity
        pos = x[:self.ndim]
        vel = x[self.ndim:]

        # f function (dynamics without control input)
        f_pos = vel
        f_vel = torch.zeros(self.ndim).to(self.device)
        f = torch.cat((f_pos, f_vel))

        # g function (control input influence)
        g = torch.zeros(2*self.ndim, self.ndim).to(self.device)
        g[self.ndim:, :] = torch.eye(self.ndim).to(self.device)

        # A matrix (df/dx)
        A = torch.zeros(2*self.ndim, 2*self.ndim).to(self.device)
        A[:self.ndim, self.ndim:] = torch.eye(self.ndim).to(self.device)

        return f, g, A
    
    def system2(self, x, u=None):
        """
        Defines the f and g functions for the double integrator system.
        x: state vector (position and velocity)
        u: control input (acceleration)
        """
        # Split state vector x into position and velocity
        print("x", x.shape)
        pos = x[:self.ndim]
        vel = x[self.ndim:2*self.ndim]
        qx, qy, qz, qw = x[2*self.ndim:]


        # f function (dynamics without control input)
        f_pos = vel
        f_vel = torch.zeros(self.ndim+4).to(self.device)
        f = torch.cat((f_pos, f_vel))
        f[5] = 9.81


        # g function (control input influence)
        g = torch.zeros(10, 4).to(self.device)
        g[3,0] = 2.0*(qx*qz + qy*qw)*self.tn/self.m
        g[4,0] = 2.0*(qy*qz - qx*qw)*self.tn/self.m
        g[5,0] = (qw*qw - qx*qx - qy*qy + qz*qz)*self.tn/self.m
        g[6,1] = 0.5*qw
        g[6,2] = -0.5*qz
        g[6,3] = 0.5*qy
        g[7,1] = 0.5*qz
        g[7,2] = 0.5*qw
        g[7,3] = -0.5*qx
        g[8,1] = -0.5*qy
        g[8,2] = 0.5*qx
        g[8,3] = 0.5*qw
        g[9,1] = -0.5*qx
        g[9,2] = -0.5*qy
        g[9,3] = -0.5*qz

        # A matrix (df/dx)
        A = torch.zeros(10, 10).to(self.device)
        A[:self.ndim, self.ndim:2*self.ndim] = torch.eye(self.ndim).to(self.device)

        return f, g, A
    