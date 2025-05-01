import os
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import run_tests

TEST_SOLUTIONS = False

if TEST_SOLUTIONS:
    from solutions.problem4 import optimal_control
    from solutions.problem4 import hamiltonian
    from solutions.problem4 import hji_vi_loss
else:
    from problem4 import optimal_control
    from problem4 import hamiltonian
    from problem4 import hji_vi_loss


print()
print('This is the first offering of this assignment, so please let us know if you find any mistakes!')
print('If you are very confident in your code implementation but do not pass the tests, please let us know and we will look into it, as it might be an issue on our side.')
print()


print('TESTING optimal_control...')
with open('tests/optimal_control_test_cases.pickle', 'rb') as f:
    optimal_control_test_cases = pickle.load(f)
run_tests(optimal_control, optimal_control_test_cases)
print()


print('TESTING hamiltonian...')
with open('tests/hamiltonian_test_cases.pickle', 'rb') as f:
    hamiltonian_test_cases = pickle.load(f)
run_tests(hamiltonian, hamiltonian_test_cases)
print()


print('TESTING hji_vi_loss...')
with open('tests/hji_vi_loss_test_cases.pickle', 'rb') as f:
    hji_vi_loss_test_cases = pickle.load(f)
run_tests(hji_vi_loss, hji_vi_loss_test_cases)
print()