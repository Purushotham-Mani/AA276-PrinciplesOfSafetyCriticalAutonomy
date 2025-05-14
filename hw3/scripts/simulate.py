import torch
import traceback
import matplotlib.pyplot as plt

from sim_utils import SamplingBasedMPC, failure_function, roll_out

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.problem3 import smooth_blending_safety_filter
else:
    from problem3 import smooth_blending_safety_filter

# environment setup
obstacles = torch.tensor([
    [1.0,  0.0, 0.5], # [px, py, radius]
    [4.0,  2.0, 1.0],
    [4.0, -2.0, 1.0],
    [7.0,  0.0, 1.5],
    [7.0,  4.0, 0.5],
    [7.0, -4.0, 0.5]
])

# task setup
goal = torch.tensor([10.0, 0.0]) # [px, py]
initial_state = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# simulation settings
nt = 1000 # number of time steps
dt = 0.01 # time step size

# mpc controller settings
mpc_nt = 100 # number of time steps for MPC optimization
mpc_ns = 1000 # number of samples for MPC optimization
mpc = SamplingBasedMPC(goal, obstacles, nt=mpc_nt, dt=dt, ns=mpc_ns, oc=1)
# NOTE: the mpc controller has an internal memory,
# so mpc should only operate on a single roll-out at a time,
# and you should call mpc.reset() before the next roll-out.
mpc.reset()

# since roll_out(.) expects a controller that operates on a batch,
# we define functions that wrap mpc that we can pass to roll_out(.).
def u_nom(x, i):
    # NOTE: mpc should only operate on a single roll-out at a time
    return mpc(x.squeeze(0)).unsqueeze(0)
def u_sb(x, i, gamma):
    # NOTE: mpc should only operate on a single roll-out at a time
    return smooth_blending_safety_filter(x.squeeze(0).clone().cpu(), mpc(x.squeeze(0)).clone().cpu(), gamma, lmbda=1e4).unsqueeze(0).to(device=x.device)

# simulate different controllers
print('Simulating nominal controller...')
mpc.reset()
xs_nom = roll_out(initial_state.unsqueeze(0).cuda(), u_nom, nt, dt, show_progress=True).squeeze(0).cpu().numpy()
gammas = [0, 0.001, 0.01]
xs_sbs = []
for gamma in gammas:
    print(f'Simulating smooth blending safety filter with gamma={gamma}...')
    try:
        mpc.reset()
        xs_sbs.append(roll_out(initial_state.unsqueeze(0).cuda(), lambda x, i: u_sb(x, i, gamma), nt, dt, show_progress=True).squeeze(0).cpu().numpy())
    except Exception as e:
        if not isinstance(e, NotImplementedError):
            traceback.print_exc()
            quit()
        print(f'Not implemented. Skipping.')
        xs_sbs.append(None)

# plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_title('13D Quadrotor')
ax.set_xlabel('$p_x$ (m)')
ax.set_ylabel('$p_y$ (m)')
# plot the obstacles
pxs = torch.linspace(-1.0, 10.0, 101)
pys = torch.linspace(-5.0, 5.0, 101)
Ps = torch.stack(torch.meshgrid(pxs, pys, indexing='ij'), dim=2)
failure_values = failure_function(Ps.reshape(-1, 2), obstacles)[1].reshape(len(pxs), len(pys))
ax.contour(pxs, pys, failure_values.T, levels=[0], colors='r')
# plot the goal
ax.scatter(goal[0], goal[1], color='green', s=20)
# plot the trajectories
ax.plot(xs_nom[:, 0], xs_nom[:, 1], label='Nominal Trajectory')
for i, gamma in enumerate(gammas):
    if xs_sbs[i] is not None:
        ax.plot(xs_sbs[i][:, 0], xs_sbs[i][:, 1], label=f'Smooth Blending Trajectory (γ={gamma})')
    else:
        print(f'Warning: not plotting γ={gamma}, since the smooth blending filter is not implemented yet.')
ax.legend()
# save the figure
plt.savefig('outputs/trajectories.png')
print('SAVED PLOT TO: outputs/trajectories.png')