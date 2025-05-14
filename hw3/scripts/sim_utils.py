import torch
from tqdm import tqdm

def control_limits():
    """
    Return a tuple (upper, lower) describing the control bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [4]
                  lower: torch float32 tensor with shape [4]
    """
    upper_limit = torch.tensor([20.0, 8.0, 8.0, 4.0])
    lower_limit = -upper_limit
    return (upper_limit, lower_limit)

def f(x):
    """
    Return the control-independent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        
    returns:
        f: torch float32 tensor with shape [batch_size, 13]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    f = torch.zeros_like(x)
    f[:, PXi] = VX
    f[:, PYi] = VY
    f[:, PZi] = VZ
    f[:, QWi] = -0.5*(WX*QX + WY*QY + WZ*QZ)
    f[:, QXi] =  0.5*(WX*QW + WZ*QY - WY*QZ)
    f[:, QYi] =  0.5*(WY*QW - WZ*QX + WX*QZ)
    f[:, QZi] =  0.5*(WZ*QW + WY*QX - WX*QY)
    f[:, VZi] = -9.8
    f[:, WXi] = -5 * WY * WZ / 9.0
    f[:, WYi] =  5 * WX * WZ / 9.0
    return f

def g(x):
    """
    Return the control-dependent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
       
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    g = torch.zeros((*x.shape, 4), device=x.device)
    g[:, VXi, 0] = 2 * (QW*QY + QX*QZ)
    g[:, VYi, 0] = 2 * (QY*QZ - QW*QX)
    g[:, VZi, 0] = (1 - 2*torch.pow(QX, 2) - 2*torch.pow(QY, 2))
    g[:, WXi:, 1:] = torch.eye(3, device=x.device)

    return g

def euler_step(x, u, dt):
    """
    Return the next states xn obtained after a discrete Euler step
    for states x, controls u, and time step dt.
    
    args:
        x: torch float32 tensor with shape  [batch_size, 13]
        u: torch float32 tensor with shape  [batch_size, 4]
        dt: float
        
    returns:
        xn: torch float32 tensor with shape [batch_size, 13]
    """
    return x + dt*(f(x) + torch.matmul(g(x), u.unsqueeze(-1)).squeeze(-1))

def roll_out(x0, u_fn, nt, dt, show_progress=False):
    """
    Return the state trajectories xts obtained by rolling out the system
    with nt discrete Euler steps using a time step of dt starting at
    states x0 and applying the controller u_fn.

    args:
        x0: torch float32 tensor with shape          [batch_size, 13]
        u_fn: Callable u=u_fn(x, i)
            u_fn takes:
                x: a torch float32 tensor with shape [batch_size, 13]
                i: an integer representing the time step
            and returns:
                u: a torch float32 tensor with shape [batch_size, 4]
        nt: int
        dt: float

    returns:
        xts: torch float32 tensor with shape         [batch_size, nt, 13]
    """
    xts = torch.zeros((len(x0), nt, 13), device=x0.device)
    for i in tqdm(range(nt), disable=not show_progress):
        xp = x0 if i == 0 else xts[:, i-1]
        xts[:, i] = euler_step(xp, u_fn(xp, i), dt)
    return xts

def failure_function(x, obstacles):
    """
    Return the index of the nearest obstacle, as well as the signed distance to the nearest obstacle.
    
    args:
        x: torch float32 tensor with shape         [batch_size, 13]
        obstacles: torch float32 tensor with shape [num_obstacles, 3]
            The obstacles in the environment,
            where the ith obstacle is represented by its position (px, py)=obstacles[i, :2] and radius r=obstacles[i, 2].

    returns:
        idx: torch int64 tensor with shape         [batch_size]
            The index of the nearest obstacle.
        dist: torch float32 tensor with shape      [batch_size]
            The signed distance to the nearest obstacle.
            Positive values indicate that the state is outside all obstacles.
            Negative values indicate that the state is inside an obstacle.
    """
    distances = torch.norm(obstacles[:, :2] - x[:, :2].unsqueeze(1), dim=2) - obstacles[:, 2]
    idx = distances.argmin(dim=1)
    dist = distances.min(dim=1).values
    return idx, dist

class SamplingBasedMPC:
    """
    A sampling-based Model Predictive Control (MPC) algorithm.

    args:
        goal: torch float32 tensor with shape      [2]
            The goal position (px, py).
        obstacles: torch float32 tensor with shape [num_obstacles, 3]
            The obstacles in the environment,
            where the ith obstacle is represented by its position (px, py)=obstacles[i, :2] and radius r=obstacles[i, 2].
        nt: int
            Number of time steps to simulate in the MPC.
        dt: float
            Time step for the simulation.
        ns: int
            Number of samples to draw for control inputs.
        ni: int
            Number of iterations for the MPC optimization.
        std: float
            Standard deviation for the control input sampling.
        oc: float
            Obstacle collision cost factor.
        gc: float
            Goal distance cost factor.
    """
    def __init__(self, goal, obstacles, nt=100, dt=0.01, ns=1000, ni=1, std=1, oc=1e3, gc=1):
        self.goal = goal
        self.obstacles = obstacles
        self.nt = nt
        self.dt = dt
        self.ns = ns 
        self.ni = ni
        self.std = std
        self.oc = oc
        self.gc = gc
        self.uu, self.lu = control_limits()
        self.us_seed = torch.tensor([9.8, 0.0, 0.0, 0.0]).repeat(nt, 1)  # initial seed control sequence
    def reset(self):
        self.us_seed = torch.tensor([9.8, 0.0, 0.0, 0.0]).repeat(self.nt, 1)  # reset seed control sequence    
    def __call__(self, x, us_seed=None):
        """
        Return the optimal control using sampling-based MPC.

        args:
            x: torch float32 tensor with shape       [13]
            us_seed: torch float32 tensor with shape [nt, 4]
                The initial seed control sequence to start the MPC optimization.
                If None, uses the seed control sequence in memory.

        returns:
            u_opt: torch float32 tensor with shape   [4]
        """
        if us_seed is None:
            us_seed = self.us_seed.clone().to(x.device)
        for i in range(self.ni):
            # Sample control sequences around the seed control sequence
            us_candidates = us_seed + self.std * torch.randn((self.ns, self.nt, 4), device=x.device)
            us_candidates = torch.clamp(us_candidates, self.lu.to(device=x.device), self.uu.to(device=x.device))
            # Roll out the system for each candidate control sequence
            xts_candidates = roll_out(x.expand(self.ns, -1), lambda x, i: us_candidates[:, i], self.nt, self.dt)
            # Compute the cost for each candidate trajectory
            ocs = self.oc * (failure_function(xts_candidates.reshape(-1, 13), self.obstacles.to(x.device))[1].reshape(self.ns, self.nt) < 0).sum(dim=1)
            gcs = self.gc * torch.norm(xts_candidates[:, :, :2] - self.goal.to(x.device), dim=2).sum(dim=1)
            # Set the seed control sequence to the one with the lowest cost
            us_seed = us_candidates[(ocs+gcs).argmin()]
        # Update the seed control sequence in memory
        self.us_seed[:-1] = us_seed[1:]  # shift the control sequence for the next iteration
        self.us_seed[-1] = torch.tensor([9.8, 0.0, 0.0, 0.0], device=self.us_seed.device)  # reset last control to a constant value
        # Return the optimal control sequence
        return us_seed[0]