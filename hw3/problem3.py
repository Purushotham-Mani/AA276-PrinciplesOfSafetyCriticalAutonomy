
import torch
import cvxpy as cp
from problem3_helper import control_limits, f, g

from problem3_helper import NeuralVF
vf = NeuralVF()

# environment setup
obstacles = torch.tensor([
    [1.0,  0.0, 0.5], # [px, py, radius]
    [4.0,  2.0, 1.0],
    [4.0, -2.0, 1.0],
    [7.0,  0.0, 1.5],
    [7.0,  4.0, 0.5],
    [7.0, -4.0, 0.5]
])

def smooth_blending_safety_filter(x, u_nom, gamma, lmbda):
    """
    Compute the smooth blending safety filter.
    Refer to the definition provided in the handout.
    You might find it useful to use functions from
    previous homeworks, which we have imported for you.
    These include:
      control_limits(.)
      f(.)
      g(.)
      vf.values(.)
      vf.gradients(.)
    NOTE: some of these functions expect batched inputs,
    but x, u_nom are not batched inputs in this case.
    
    args:
        x:      torch tensor with shape [13]
        u_nom:  torch tensor with shape [4]
        
    returns:
        u_sb:   torch tensor with shape [4]
    """
    # YOUR CODE HERE
    x_batched = x.unsqueeze(0)
    fx = f(x_batched)[0].detach().cpu().numpy()
    gx = g(x_batched)[0].detach().cpu().numpy()
    u_nom_np = u_nom.detach().cpu().numpy()

    # Compute minimum value function and corresponding gradient over all obstacles
    min_value = float('inf')
    min_grad = None

    for obs in obstacles:
        px, py, orad = obs
        x_hat = x.clone()
        x_hat[0] = (orad / 0.5)*(x[0] - px)  
        x_hat[1] = (orad / 0.5)*(x[1] - py)
        x_hat[7] = (orad / 0.5)*x_hat[7]
        x_hat[8] = (orad / 0.5)*x_hat[8]
        x_hatbatched = x_hat.unsqueeze(0)

        v = vf.values(x_hatbatched)[0].item()

        if v < min_value:
            min_value = v
            grad = vf.gradients(x_hatbatched)[0]
            grad_scaled = grad
            min_grad = grad_scaled.numpy()

    # Setup QP
    u = cp.Variable(4)
    s = cp.Variable(1)

    obj = cp.Minimize(cp.sum_squares(u - u_nom_np) + lmbda * (s**2))

    u_upper, u_lower = control_limits()

    constraints = [
        s >= 0,
        min_grad @ (fx + gx @ u) + gamma * min_value + s >= 0,
        u >= u_lower.numpy(),
        u <= u_upper.numpy()
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve()
    u_sb = u
    print(prob.status)
    if prob.status != "optimal":
        return u_nom

    return torch.tensor(u_sb.value, dtype=torch.float32) # NOTE: ensure you return a float32 tensor
