import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from plot_utils import failure_mask, roll_out

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.problem4 import optimal_control
else:
    from problem4 import optimal_control
from problem4_helper import NeuralVF
neuralvf = NeuralVF(ckpt_path='outputs/vf.ckpt')

fig, ax = plt.subplots()
ax.set_title('$V(x)$ for x=(., ., 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0)')
ax.set_xlabel('$p_x$ (m)')
ax.set_ylabel('$p_y$ (m)')
px = torch.linspace(-3, 3, 100)
py = torch.linspace(-3, 3, 100)
slice = torch.tensor([
    0., 0., 0.,
    1., 0., 0., 0.,
    5., 0., 0.,
    0., 0., 0.
])

print('creating plot...')
# values
PX, PY = torch.meshgrid(px, py, indexing='ij')
X = torch.zeros((len(px), len(py), 13))
X[..., 0] = PX
X[..., 1] = PY
X[..., 2:] = slice[2:]
V = neuralvf.values(X.reshape(-1, 13)).reshape(len(px), len(py))
px = px.detach().cpu().numpy()
py = py.detach().cpu().numpy()
V = V.detach().cpu().numpy()
vbar=3
im = ax.pcolormesh(px, py, V.T, cmap='RdBu', vmin=-vbar, vmax=vbar)
fig.colorbar(im)
ax.contour(px, py, V.T, colors='k', levels=[0])
ax.contour(px, py, X[..., :2].norm(dim=-1)-0.5, colors='r', levels=[0])
# trajectories
state_min, state_max = torch.clone(slice), torch.clone(slice)
state_min[0], state_min[1] = -5, -1
state_max[0], state_max[1] = -2, 1
x0 = torch.rand(100, 13)*(state_max-state_min)+state_min
is_safe = neuralvf.values(x0) > 0
nt = 100
dt = 0.01
u_fn = lambda x: optimal_control(x, neuralvf.gradients(x).detach())
xts = roll_out(x0, u_fn, nt, dt)
for i, xt in enumerate(xts):
    ax.plot(xt[:, 0], xt[:, 1], color='green' if is_safe[i] else 'orange')
safe_line = mlines.Line2D([], [], color='green', label='marked safe')
unsafe_line = mlines.Line2D([], [], color='orange', label='marked unsafe')
ax.legend(handles=[safe_line, unsafe_line])
plt.savefig('outputs/plot.png')
plt.close()
print('PLOT SAVED TO outputs/plot.png')

is_fail = torch.any(failure_mask(xts.reshape(-1, 13)).reshape(len(x0), nt), dim=1)
false_safety_rate = (torch.sum(is_fail[is_safe])/torch.sum(is_safe)).item()
print(f'false safety rate: {false_safety_rate}')