import torch
import matplotlib.pyplot as plt
from neural_clbf.controllers import NeuralCBFController

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.part3 import plot_h, plot_and_eval_xts
    from solutions.part1 import state_limits
else:
    from part3 import plot_h, plot_and_eval_xts
    from part1 import state_limits
state_max, state_min = state_limits()

ckptpath = 'outputs/cbf.ckpt'
neural_controller = NeuralCBFController.load_from_checkpoint(ckptpath)

fig, ax = plt.subplots()
ax.set_title('$h(x)$ for x=(., ., 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0)')
ax.set_xlabel('$p_x$ (m)')
ax.set_ylabel('$p_y$ (m)')
px = torch.linspace(-3, 3, 10)
py = torch.linspace(-3, 3, 10)
slice = torch.tensor([
    0., 0., 0.,
    1., 0., 0., 0.,
    5., 0., 0.,
    0., 0., 0.
])
h_fn = lambda x: -neural_controller.V_with_jacobian(x)[0]
dhdx_fn = lambda x: -neural_controller.V_with_jacobian(x)[1].squeeze(1)
# restrict px, py
state_max[0], state_min[0] = -2, -5
state_max[1], state_min[1] = 1, -1
x0 = torch.rand(100, 13)*(state_max-state_min)+state_min
x0[:, 2:] = slice[2:]
def u_ref_fn(x):
    u = torch.zeros((len(x), 4))
    u[:, 0] = 9.8
    return u
gamma = 0
lmbda = 1e4
nt = 10
dt = 0.1

print('running plot_h...')
plot_h(fig, ax, px, py, slice, h_fn)

print('running plot_and_eval_xts...')
false_safety_rate = plot_and_eval_xts(fig, ax, x0, u_ref_fn, h_fn, dhdx_fn, gamma, lmbda, nt, dt)

plt.savefig('outputs/plot.png')
plt.close()
print('plot saved to outputs/plot.png')
print(f'false safety rate: {false_safety_rate}')