import os
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import run_tests

TEST_SOLUTIONS = False

if TEST_SOLUTIONS:
    from solutions.part2 import euler_step
    from solutions.part2 import roll_out
    from solutions.part2 import u_qp
else:
    from part2 import euler_step
    from part2 import roll_out
    from part2 import u_qp


print()
print('This is the first offering of this assignment, so please let us know if you find any mistakes!')
print('If you are very confident in your code implementation but do not pass the tests, please let us know and we will look into it, as it might be an issue on our side.')
print()


print('TESTING euler_step...')
with open('tests/part2/euler_step_test_cases.pickle', 'rb') as f:
    euler_step_test_cases = pickle.load(f)
run_tests(euler_step, euler_step_test_cases)
print()


import torch
def u_fn(x):
    u = torch.zeros((len(x), 4))
    u[:, 0] = 9.8
    u[:, 1] = 1
    u[:, 2] = 1
    u[:, 3] = 1
    return u


print('TESTING roll_out...')
with open('tests/part2/roll_out_test_cases.pickle', 'rb') as f:
    roll_out_test_cases = pickle.load(f)
    for case in roll_out_test_cases:
        case['args'] = tuple([case['args'][i] if i != 1 else u_fn for i in range(len(case['args']))])
run_tests(roll_out, roll_out_test_cases)
print()


print('TESTING u_qp...')
with open('tests/part2/u_qp_test_cases.pickle', 'rb') as f:
    u_qp_test_cases = pickle.load(f)
run_tests(u_qp, u_qp_test_cases)
print()