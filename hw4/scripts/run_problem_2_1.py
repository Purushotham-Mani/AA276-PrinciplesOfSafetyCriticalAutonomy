import numpy as np
np.random.seed(0)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dynamics import f_disturbed
from utils.simulation import compete
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.problem_2_1 import ControllerA, ControllerB
else:
    from problem_2_1 import ControllerA, ControllerB

print('Save animation? (y/n): ', end='')
save_animation = input() == 'y'

print('Running A...')
s0 = [0.0, np.pi, 0.0, 0.0, -1.]
controller = ControllerA()
dt = 0.01
nt = 1000
f = lambda s, u: f_disturbed(s, u, sigma=0)
animation_save_path = 'outputs/problem_2_1_A.gif' if not USE_SOLUTIONS \
    else 'solutions/outputs/problem_2_1_A.gif'
controller.reset()
qualified_A, max_distance_A = compete(s0, controller, dt, nt, f,
                                  save_animation, animation_save_path)

print('Running B...')
s0 = [0.0, np.pi, 0.0, 0.0, 1.]
controller = ControllerB()
dt = 0.01
nt = 1000
f = lambda s, u: f_disturbed(s, u, sigma=0)
animation_save_path = 'outputs/problem_2_1_B.gif' if not USE_SOLUTIONS \
    else 'solutions/outputs/problem_2_1_B.gif'
controller.reset()
qualified_B, max_distance_B = compete(s0, controller, dt, nt, f,
                                  save_animation, animation_save_path)

print()
print('###############')
print('### SUMMARY ###')
print('###############')
print()
print('A:')
print(f'Qualified:        {qualified_A}')
print(f'Max distance (m): {max_distance_A}')
print()
print('B:')
print(f'Qualified:        {qualified_B}')
print(f'Max distance (m): {max_distance_B}')