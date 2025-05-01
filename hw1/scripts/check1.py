import os
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import run_tests

TEST_SOLUTIONS = False

if TEST_SOLUTIONS:
    from solutions.part1 import state_limits
    from solutions.part1 import control_limits
    from solutions.part1 import safe_mask
    from solutions.part1 import failure_mask
    from solutions.part1 import g
else:
    from part1 import state_limits
    from part1 import control_limits
    from part1 import safe_mask
    from part1 import failure_mask
    from part1 import g


print()
print('This is the first offering of this assignment, so please let us know if you find any mistakes!')
print('If you are very confident in your code implementation but do not pass the tests, please let us know and we will look into it, as it might be an issue on our side.')
print()


print('TESTING state_limits...')
with open('tests/part1/state_limits_test_cases.pickle', 'rb') as f:
    state_limits_test_cases = pickle.load(f)
run_tests(state_limits, state_limits_test_cases)
print()


print('TESTING control_limits...')
with open('tests/part1/control_limits_test_cases.pickle', 'rb') as f:
    control_limits_test_cases = pickle.load(f)
run_tests(control_limits, control_limits_test_cases)
print()


print('TESTING safe_mask...')
with open('tests/part1/safe_mask_test_cases.pickle', 'rb') as f:
    safe_mask_test_cases = pickle.load(f)
run_tests(safe_mask, safe_mask_test_cases)
print()


print('TESTING failure_mask...')
with open('tests/part1/failure_mask_test_cases.pickle', 'rb') as f:
    failure_mask_test_cases = pickle.load(f)
run_tests(failure_mask, failure_mask_test_cases)
print()


print('TESTING g...')
with open('tests/part1/g_test_cases.pickle', 'rb') as f:
    g_test_cases = pickle.load(f)
run_tests(g, g_test_cases)
print()