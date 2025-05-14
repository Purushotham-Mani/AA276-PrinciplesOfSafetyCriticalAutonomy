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
        x_hat[0] = x[0] - px  # shift px
        x_hat[1] = x[1] - py  # shift py
        x_hatbatched = x_hat.unsqueeze(0)

        v = vf.values(x_hatbatched)[0].item()
        v_scaled = (orad / 0.5) * v

        if v_scaled < min_value:
            min_value = v_scaled
            grad = vf.gradients(x_hatbatched)[0]
            grad_scaled = (orad / 0.5) * grad
            min_grad = grad_scaled.numpy()

    # Setup QP
    u = cp.Variable(4)
    s = cp.Variable(1)

    obj = cp.Minimize(cp.norm(u - u_nom_np) + lmbda * s)

    u_lower, u_upper = control_limits()

    constraints = [
        s >= 0,
        min_grad @ (fx + gx @ u) + gamma * min_value + s >= 0,
        u >= u_lower.numpy(),
        u <= u_upper.numpy()
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve()
    u_sb = u
    return torch.tensor(u_sb.value, dtype=torch.float32) # NOTE: ensure you return a float32 tensor