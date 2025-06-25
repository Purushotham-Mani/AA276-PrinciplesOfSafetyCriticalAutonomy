import numpy as np
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.animation import animate

def simulate(s0, u_fn, dt, nt, f):
    """Simulate the cart-pole system.

    Args:
        s0 (np.ndarray): The initial state with shape (n)
        u_fn (callable): The control function u_fn(s, t) -> u with shape (m)
        dt (float): The time step
        nt (int): The number of time steps
        f (callable): The system dynamics f(s, u) -> dynamics with shape (n)

    Returns:
        s_history (np.ndarray): The state history with shape (nt, n)
        u_history (np.ndarray): The control history with shape (nt, m)
        t_history (np.ndarray): The time history with shape (nt)
    """
    s_history = np.full((nt, len(s0)), np.nan)
    u_history = np.full((nt, 1), np.nan)
    t_history = np.full((nt,), np.nan)
    s_history[0] = s0
    for i in tqdm(range(nt), desc='Simulating...'):
        u = u_fn(s_history[i, :4], i*dt)
        u_history[i] = u
        t_history[i] = i*dt
        if i < nt-1:
            s_history[i+1] = s_history[i] + dt*f(s_history[i], u)
    return s_history, u_history, t_history

def compete(s0, controller, dt, nt, f, save_animation, animation_save_path):
    """Compete in the cart-pole race.

    Args:
        s0 (np.ndarray): The initial state with shape (n)
        controller (Controller): The controller with functions
            controller.reset()
            controller.u_fn(s, t) -> u with shape (m)
            controller.data_to_visualize() -> data_to_visualize (dict)
        dt (float): The time step
        nt (int): The number of time steps
        f (callable): The system dynamics f(s, u) -> dynamics with shape (n)
        save_animation (bool): Whether to animate the simulation
        animation_save_path (str): The path to save the animation

    Returns:
        qualified (bool): Whether the run has qualified
        max_distance (float): The maximum distance reached before a safety violation
    """
    # simulate
    controller.reset()
    s_history, u_history, t_history = simulate(
         s0, controller.u_fn, dt, nt, f)
    print('Done.')

    # animate
    if save_animation:
        skip_frames = int(0.1/dt) # number of frames to skip in the animation
        fig, ani = animate(
             t_history,
             s_history[:, 0],
             s_history[:, 1],
             d=s_history[:, 4] if s_history.shape[-1] > 4 else None,
             extra=controller.data_to_visualize(),
            skip_frames=skip_frames)
        with tqdm(total=t_history.size, desc="Animating...") as pbar:
                ani.save(animation_save_path, writer='ffmpeg', progress_callback=lambda i, n: pbar.update(skip_frames))
        print(f'Done. Animation saved to {animation_save_path}')

    # score
    print('Scoring...')
    has_fallen = np.abs(s_history[:, 1]-np.pi) >= np.pi/2
    if np.any(has_fallen):
        qualified = False
        print(f'Run has been DISQUALIFIED!')
        first_fall_index = np.argmax(has_fallen).item()
        print(f'First fall at time: {t_history[first_fall_index]:.2f} (s)')
        max_distance = np.max(s_history[:first_fall_index, 0])
        print(f'Max distance before fall: {max_distance:.3f} (m)')
    else:
        qualified = True
        print(f'Run has QUALIFIED!')
        max_distance = np.max(s_history[:, 0])
        print(f'Max distance: {max_distance:.3f} (m)')

    return qualified, max_distance