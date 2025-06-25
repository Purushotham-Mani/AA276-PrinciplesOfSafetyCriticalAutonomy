import numpy as np
np.random.seed(0)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dynamics import f_disturbed
from utils.simulation import compete
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.problem_2_2 import Controller
else:
    from problem_2_2 import Controller

print('Save animation? (y/n): ', end='')
save_animation = input() == 'y'

ds = [-5., -2.5, 0., 2.5, 5.]
labels = ['A', 'B', 'C', 'D', 'E']
qualifieds = []
max_distances = []

controller = Controller()
for i in range(len(ds)):
    print(f'Running {labels[i]}...')
    s0 = [0.0, np.pi, 0.0, 0.0, ds[i]]
    dt = 0.01
    nt = 1000
    f = lambda s, u: f_disturbed(s, u, sigma=5)
    animation_save_path = f'outputs/problem_2_2_{labels[i]}.gif' if not USE_SOLUTIONS \
        else f'solutions/outputs/problem_2_2_{labels[i]}.gif'
    controller.reset()
    qualified, max_distance = compete(s0, controller, dt, nt, f,
                                      save_animation, animation_save_path)
    qualifieds.append(qualified)
    max_distances.append(max_distance)

print()
print('###############')
print('### SUMMARY ###')
print('###############')
print()
for i in range(len(ds)):
    print(f'{labels[i]}:')
    print(f'Qualified:        {qualifieds[i]}')
    print(f'Max distance (m): {max_distances[i]}')
    print()
print(f'All qualified:            {np.all(qualifieds)}')
print(f'Average max distance (m): {np.mean(max_distances)}')