"""
AA 276 Homework 1 | Coding Portion | Part 2 of 3


OVERVIEW

In this file, you will implement functions for simulating the
13D quadrotor system discretely and computing the CBF-QP controller.


INSTRUCTIONS

Make sure you pass the tests for Part 1 before you begin.
Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check2.py`.
"""


import torch
from part1 import f, g


"""Note: the following functions operate on batched inputs."""


def euler_step(x, u, dt):
    """
    Return the next states xn obtained after a discrete Euler step
    for states x, controls u, and time step dt.
    Hint: we have imported f(x) and g(x) from Part 1 for you to use.
    
    args:
        x: torch float32 tensor with shape [batch_size, 13]
        u: torch float32 tensor with shape [batch_size, 4]
        dt: float
        
    returns:
        xn: torch float32 tensor with shape [batch_size, 13]
    """
    dx = f(x) + torch.bmm(g(x), u.unsqueeze(-1)).squeeze(-1)
    xn = x + dx * dt
    return xn

    
def roll_out(x0, u_fn, nt, dt):
    """
    Return the state trajectories xts obtained by rolling out the system
    with nt discrete Euler steps using a time step of dt starting at
    states x0 and applying the controller u_fn.
    Note: The returned state trajectories should start with x1; i.e., omit x0.
    Hint: You should use the previous function, euler_step(x, u, dt).

    args:
        x0: torch float32 tensor with shape [batch_size, 13]
        u_fn: Callable u=u_fn(x)
            u_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size, 4]
        nt: int
        dt: float

    returns:
        xts: torch float32 tensor with shape [batch_size, nt, 13]
    """
    xts = torch.zeros((x0.shape[0], nt, x0.shape[1]), dtype=torch.float32, device=x0.device)
    xc = x0.clone()
    for i in range(nt):
        u = u_fn(xc)
        xc = euler_step(xc, u, dt)
        xts[:, i, :] = xc
    return xts


import cvxpy as cp
from part1 import control_limits
import numpy as np

def u_qp(x, h, dhdx, u_ref, gamma, lmbda):
    """
    Return the solution of the CBF-QP with parameters gamma and lmbda
    for the states x, CBF values h, CBF gradients dhdx, and reference controls u_nom.
    Hint: consider using CVXPY to solve the optimization problem: https://www.cvxpy.org/version/1.2/index.html
        Note: We are using an older version of CVXPY (1.2.1) to use the neural CBF library.
            Make sure you are looking at the correct version of documentation.
        Note: You may want to use control_limits() from Part 1.
    Hint: If you use multiple libraries, make sure to properly handle data-type conversions.
        For example, to safely convert a torch tensor to a numpy array: x = x.detach().cpu().numpy()

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        h: torch float32 tensor with shape [batch_size]
        dhdx: torch float32 tensor with shape [batch_size, 13]
        u_ref: torch float32 tensor with shape [batch_size, 4]
        gamma: float
        lmbda: float

    returns:
        u_qp: torch float32 tensor with shape [batch_size, 4]
    """
    batch_size = x.shape[0]

    u = cp.Variable((batch_size, 4))
    delta = cp.Variable((batch_size))

    fx = f(x).detach().cpu().numpy()
    gx = g(x).detach().cpu().numpy()

    u_ref = u_ref.detach().cpu().numpy()
    h = h.detach().cpu().numpy()
    dhdx = dhdx.detach().cpu().numpy()
    upper, lower = control_limits()    
    upper = upper.detach().cpu().numpy()
    lower = lower.detach().cpu().numpy()

    objective = cp.Minimize(cp.sum_squares(u - u_ref)+ lmbda* cp.sum_squares(delta))
    constraints = []
    for b in range(batch_size):
        constraints += [dhdx[b] @ (fx[b] + gx[b] @ u[b]) + gamma * h[b] + delta[b]>=0]
        constraints += [u[b] <= upper, u[b] >= lower]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    u_qp = torch.tensor(u.value, dtype=torch.float32, device=x.device)

    return u_qp
