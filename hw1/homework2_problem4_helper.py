"""
Helper functions to query the learned values and gradients of cbf.ckpt.
NOTE: To use NeuralCBF defined below, you need to have your solution files from Homework 1
in this directory. Make sure your hw1 venv is activated to use NeuralCBF.
"""

"""
Helper class for querying cbf.ckpt.
neuralcbf = NeuralCBF()
h_values = neuralcbf.h_values(x)
h_gradients = neuralcbf.h_gradients(x)
"""
class NeuralCBF:
    def __init__(self, ckpt_path='outputs/cbf.ckpt'):
        try:
            from neural_clbf.controllers import NeuralCBFController
        except Exception as e:
            print(str(e))
            print('MAKE SURE YOU HAVE THE VENV FROM HW1 ACTIVATED')
        try:
            self.model = NeuralCBFController.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(str(e))
            print('MAKE SURE YOUR FILES FROM HOMEWORK 1 ARE IN THE SAME DIRECTORY AS THIS FILE')

    def values(self, x):
        """
        args:
            x: torch tensor with shape    [batch_size, 13]
        
        returns:
            h(x): torch tensor with shape [batch_size]
        """
        return -self.model.V_with_jacobian(x)[0]
    
    def gradients(self, x):
        """
        args:
            x: torch tensor with shape       [batch_size, 13]

        returns:
            dhdx(x): torch tensor with shape [batch_size, 13]
        """
        return -self.model.V_with_jacobian(x)[1].squeeze(1)