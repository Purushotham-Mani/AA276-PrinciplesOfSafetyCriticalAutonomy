import torch

def control_limits():
    """
    Return a tuple (upper, lower) describing the control bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [4]
                  lower: torch float32 tensor with shape [4]
    """
    upper_limit = torch.tensor([20.0, 8.0, 8.0, 4.0])
    lower_limit = -upper_limit
    return (upper_limit, lower_limit)

def f(x):
    """
    Return the control-independent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        
    returns:
        f: torch float32 tensor with shape [batch_size, 13]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    f = torch.zeros_like(x)
    f[:, PXi] = VX
    f[:, PYi] = VY
    f[:, PZi] = VZ
    f[:, QWi] = -0.5*(WX*QX + WY*QY + WZ*QZ)
    f[:, QXi] =  0.5*(WX*QW + WZ*QY - WY*QZ)
    f[:, QYi] =  0.5*(WY*QW - WZ*QX + WX*QZ)
    f[:, QZi] =  0.5*(WZ*QW + WY*QX - WX*QY)
    f[:, VZi] = -9.8
    f[:, WXi] = -5 * WY * WZ / 9.0
    f[:, WYi] =  5 * WX * WZ / 9.0
    return f

def g(x):
    """
    Return the control-dependent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
       
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    g = torch.zeros((*x.shape, 4), device=x.device)
    g[:, VXi, 0] = 2 * (QW*QY + QX*QZ)
    g[:, VYi, 0] = 2 * (QY*QZ - QW*QX)
    g[:, VZi, 0] = (1 - 2*torch.pow(QX, 2) - 2*torch.pow(QY, 2))
    g[:, WXi:, 1:] = torch.eye(3, device=x.device)

    return g

"""
Helper class for querying vf.ckpt.
neuralvf = NeuralVF()
values = neuralvf.values(x)
gradients = neuralvf.gradients(x)
"""
class NeuralVF:
    def __init__(self, ckpt_path='outputs/vf.ckpt'):
        import os
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

        from libraries.DeepReach_MPC.utils import modules
        from libraries.DeepReach_MPC.dynamics.dynamics import Quadrotor

        dynamics = Quadrotor(collisionR=0.5, collective_thrust_max=20, set_mode='avoid')
        model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type='sine', mode='mlp',
                                    final_layer_factor=1., hidden_features=512, num_hidden_layers=3, 
                                    periodic_transform_fn=dynamics.periodic_transform_fn)
        model.cuda()
        model.load_state_dict(torch.load(ckpt_path)['model'])

        self.dynamics = dynamics
        self.model = model

    def values(self, x):
        """
        args:
            x: torch tensor with shape      [batch_size, 13]
        returns:
            values: torch tensor with shape [batch_size]
        """
        coords = torch.concatenate((torch.ones((len(x), 1), device=x.device), x), dim=1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            model_results = self.model({'coords': model_input.cuda()})
        values = self.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].detach().squeeze(dim=-1))
        return values.cpu()
    
    def gradients(self, x):
        """
        args:
            x: torch tensor with shape         [batch_size, 13]
        returns:
            gradients: torch tensor with shape [batch_size, 13]
        """
        coords = torch.concatenate((torch.ones((len(x), 1), device=x.device), x), dim=1)
        model_input = self.dynamics.coord_to_input(coords)
        model_results = self.model({'coords': model_input.cuda()})
        gradients = self.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))[:, 1:]
        return gradients.detach().cpu()