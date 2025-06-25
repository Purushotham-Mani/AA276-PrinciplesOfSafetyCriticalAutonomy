"""
Adapted from Autonomous Systems Lab (ASL), Stanford University
"""

import numpy as np

def f(s: np.ndarray, u: np.ndarray, mp=1, mc=1, L=1, g=2., u_bar=10) -> np.ndarray:
    """Compute the cart-pole state derivative

    Args:
        s (np.ndarray): The cartpole state: [x, theta, x_dot, theta_dot], shape (4)
        u (np.ndarray): The cartpole control: [F_x], shape (1)

    Returns:
        np.ndarray: The state derivative, shape (4)
    """
    x, θ, dx, dθ = s
    u = np.clip(u, -u_bar, u_bar)
    sinθ, cosθ = np.sin(θ), np.cos(θ)
    h = mc + mp * (sinθ**2)
    ds = np.array(
        [
            dx,
            dθ,
            (mp * sinθ * (L * (dθ**2) + g * cosθ) + u[0]) / h,
            -((mc + mp) * g * sinθ + mp * L * (dθ**2) * sinθ * cosθ + u[0] * cosθ)
            / (h * L),
        ]
    )
    return ds

def f_disturbed(s: np.ndarray, u: np.ndarray, mp=1, mc=1, L=1, g=2., u_bar=10, sigma=5) -> np.ndarray:
    """Compute the cart-pole state derivative

    Args:
        s (np.ndarray): The cartpole state: [x, theta, x_dot, theta_dot, disturbance], shape (5)
        u (np.ndarray): The cartpole control: [F_x], shape (1)

    Returns:
        np.ndarray: The state derivative, shape (5)
    """
    x, θ, dx, dθ, disturbance = s
    u = np.clip(u, -u_bar, u_bar)
    sinθ, cosθ = np.sin(θ), np.cos(θ)
    h = mc + mp * (sinθ**2)
    ds = np.array(
        [
            dx,
            dθ,
            (mp * sinθ * (L * (dθ**2) + g * cosθ) + u[0]) / h,
            -((mc + mp) * g * sinθ + mp * L * (dθ**2) * sinθ * cosθ + u[0] * cosθ)
            / (h * L) + disturbance,
            np.random.normal(0, sigma),
        ]
    )
    if disturbance < -5:
        ds[4] = max(ds[4], 0)
    if disturbance > 5:
        ds[4] = min(ds[4], 0)
    return ds