import numpy as np
np.random.seed(0)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dynamics import f
from utils.simulation import compete
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.problem_1_3 import Controller
else:
    from problem_1_3 import Controller

s0 = [0.0, np.pi, 0.0, 0.0]
controller = Controller()
dt = 0.01
nt = 1000
f = f
print('Save animation? (y/n): ', end='')
save_animation = input() == 'y'
animation_save_path = 'outputs/problem_1_3.gif' if not USE_SOLUTIONS \
    else 'solutions/outputs/problem_1_3.gif'

controller.reset()
qualified, max_distance = compete(s0, controller, dt, nt, f,
                                  save_animation, animation_save_path)

print()
print('###############')
print('### SUMMARY ###')
print('###############')
print()
print(f'Qualified:        {qualified}')
print(f'Max distance (m): {max_distance}')