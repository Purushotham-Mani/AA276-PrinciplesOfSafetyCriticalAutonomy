import torch

"""For your convenience, we have provided f(x) from Homework 1."""
def f(x):
    """
    Return the control-independent part of the control-affine dynamics for the 13D quadrotor system.

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

"""For your convenience, we have provided g(x) from Homework 1."""
def g(x):
    """
    Return the control-dependent part of the control-affine dynamics for the 13D quadrotor system.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
       
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    g = torch.zeros((*x.shape, 4))
    g[:, VXi, 0] = 2 * (QW*QY + QX*QZ)
    g[:, VYi, 0] = 2 * (QY*QZ - QW*QX)
    g[:, VZi, 0] = (1 - 2*torch.pow(QX, 2) - 2*torch.pow(QY, 2))
    g[:, WXi:, 1:] = torch.eye(3)

    return g

"""IMPLEMENT THE FOLLOWING FUNCTIONS"""

def optimal_control(x, dVdx):
    """
    Compute the optimal safety controller for the 13D quadrotor system for
    states x and
    value function spatial gradients dVdx.

    args:
        x:     torch tensor with shape [batch_size, 13]
        dVdx:  torch tensor with shape [batch_size, 13]
    
    returns:
        u_opt: torch tensor with shape [batch_size, 4]
    """
    u_low = torch.tensor([-20.0, -8.0, -8.0, -4.0], dtype=torch.float32, device=x.device)
    u_high = torch.tensor([20.0, 8.0, 8.0, 4.0], dtype=torch.float32, device=x.device)
    
    h = torch.bmm(dVdx.unsqueeze(1), g(x)).squeeze(1)  # [batch_size, 4]
    u_opt = torch.where( h < 0, u_low, u_high)

    return u_opt

def hamiltonian(x, dVdx):
    """
    Compute the Hamiltonian for the 13D quadrotor system for
    states x and
    value function spatial gradients dVdx.

    args:
        x:    torch tensor with shape [batch_size, 13]
        dVdx: torch tensor with shape [batch_size, 13]

    returns:
        ham:  torch tensor with shape [batch_size]
    """
    f_ = f(x) + torch.bmm(g(x), optimal_control(x, dVdx).unsqueeze(2)).squeeze(2)
    ham = torch.sum(dVdx * f_, dim=1)
    return ham

def hji_vi_loss(x, l, V, dVdt, dVdx):
    """
    Compute the HJI-VI loss term for the
    states x,
    failure function values l,
    value function values V,
    value function temporal gradients dVdt, and
    value function spatial gradients dVdx.
    This corresponds to h2(.) in Equation (14)
    in the DeepReach paper: https://arxiv.org/pdf/2011.02082.
    NOTE: You should return a batch of losses, i.e., you can
    interpret the ||.|| as |.| in Equation (14).

    args:
        x:    torch tensor with shape [batch_size, 13]
        l:    torch tensor with shape [batch_size]
        V:    torch tensor with shape [batch_size]
        dVdt: torch tensor with shape [batch_size]
        dVdx: torch tensor with shape [batch_size, 13] 

    returns:
        h2:   torch tensor with shape [batch_size]
    """
    ham = hamiltonian(x, dVdx)
    
    # Compute the HJI-VI loss term
    h2 = torch.abs(torch.minimum(dVdt + ham, l- V))
    
    return h2