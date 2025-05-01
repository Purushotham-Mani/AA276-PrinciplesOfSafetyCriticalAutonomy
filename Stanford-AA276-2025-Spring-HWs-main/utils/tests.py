# test_cases = [
#     {
#         'args': (1, 2),
#         'kwargs': {'c': 3},
#         'expected': 6
#     },
#     {
#         'args': (),
#         'kwargs': {'a': 4, 'b': 5, 'c': 6},
#         'expected': 15
#     }
# ]

def make_tests(func, test_cases):
    """Populates the 'expected' entry in the test case dictionaries, so that run_tests can be used subsequently."""
    for i, case in enumerate(test_cases):
        case['expected'] = func(*case['args'], **case['kwargs'])

def run_tests(func, test_cases):
    for i, case in enumerate(test_cases):
        result = func(*case['args'], **case['kwargs'])

        def check(result, expected):
            assert type(result) == type(expected), f'Test {i} failed due to type mismatch: got {type(result)}, expected {type(expected)}'

            # different ways to verify result, based on type(result)
            
            import torch
            if isinstance(result, torch.Tensor):
                assert result.dtype == expected.dtype, f'Test {i} failed due to dtype mismatch: got {result.dtype}, expected {expected.dtype}'
                assert torch.allclose(result, expected, rtol=1e-4), f'Test {i} failed due to value mismatch'

            elif isinstance(result, int) or isinstance(result, bool):
                assert result == expected, f'Test {i} failed due to value mismatch'

            elif isinstance(result, tuple) or isinstance(result, list):
                assert len(result) == len(expected), f'Test {i} failed due to length mismatch: got {len(result)}, expected {len(expected)}'
                for j in range(len(result)):
                    check(result[j], expected[j])
            
            else:
                raise NotImplementedError

        check(result, case['expected'])
        print(f'Test {i} passed.')