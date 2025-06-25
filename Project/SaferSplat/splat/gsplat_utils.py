
import json
import torch
from pathlib import Path
import open3d as o3d
import time

from ellipsoids.mesh_utils import create_gs_mesh
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
from ellipsoids.covariance_utils import compute_cov1
from splat.distances import distance_point_ellipsoid, batch_point_distance, batch_squared_point_distance, batch_mahalanobis_distance
from ns_utils.nerfstudio_utils import GaussianSplat, SH2RGB
from scipy.stats import chi2
import os
import math

class GSplatLoader():
    def __init__(self, gsplat_location, device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.workspace_path = Path(__file__).parent.parent.parent / 'gsplats' / 'workspace'
        if isinstance(gsplat_location, Path):
            self.load_gsplat_from_nerfstudio(gsplat_location)
        else:
            raise ValueError('GSplat file must be either a .json or .yml file.')
        
    def load_gsplat_from_nerfstudio(self, gsplat_location):

        cwd = os.getcwd()

        os.chdir(self.workspace_path)
        self.splat = GaussianSplat(gsplat_location,
                    test_mode= "inference",
                    dataset_mode = 'train',
                    device = self.device)
        os.chdir(cwd)

        self.means = self.splat.pipeline.model.means.detach().clone()
        self.rots = self.splat.pipeline.model.quats.detach().clone()
        self.Rnerf_world = torch.tensor([[1,0,0],[0,0,1],[0,1,0]], dtype=torch.float32, device=self.device)
        self.means = torch.matmul(self.Rnerf_world, self.means.T).T

        rots = quaternion_to_rotation_matrix(self.rots)
        self.rots = torch.matmul(self.Rnerf_world[None, :, :], rots)
        # chi2_val = math.sqrt(chi2.ppf(0.95, df=3))
        self.scales = self.splat.pipeline.model.scales.detach().clone()
        self.scales = torch.exp(self.scales)#*chi2_val
        # print("1st mean:", self.means[0])
        self.covs_inv = compute_cov1(self.rots, 1 / self.scales)
        self.covs = compute_cov1(self.rots, self.scales)

        self.colors = SH2RGB(self.splat.pipeline.model.features_dc.detach().clone())
        self.opacities = torch.sigmoid(self.splat.pipeline.model.opacities.detach().clone())

        self.bump_center = torch.tensor([1.5, -8.0, -1.0], device=self.device)
        self.bump_radius = 1.0

        
        print(f'There are {self.means.shape} Gaussians in the GSplat model')
        # print("2nd mean:", self.means[0])
        return

    #NOTE: Need to provide the robot radius OR the robot R and S matrices
    def smooth_bump_epsilon(self, x, eps_max=2.0):
        """
        Args:
            x: Tensor of shape (..., 3), points at which to evaluate
            center: Tensor of shape (3,), the center of the bump
            radius: Scalar, radius of the bump region
            eps_max: Maximum value of epsilon inside the bump

        Returns:
            epsilon: (...,), scalar value at each point
            grad: (..., 3), gradient w.r.t. x
            hess: (..., 3, 3), Hessian w.r.t. x
        """
        delta = x - self.bump_center
        dist_sq = torch.sum(delta**2, dim=-1, keepdim=True)
        r_sq = self.bump_radius**2
        inside = dist_sq < r_sq

        # Calculate bump only where inside the ball
        z = dist_sq / r_sq  # (..., 1)
        one_minus_z = 1 - z
        inv = 1. / one_minus_z  # (..., 1)
        e = eps_max * torch.exp(-inv)

        epsilon = torch.where(inside, e, torch.zeros_like(e[..., 0]))

        # Gradient
        de_dz = -e * inv**2  # (..., 1)
        dz_dx = 2 * delta / r_sq  # (..., 3)
        grad = torch.where(
            inside,
            de_dz * dz_dx,  # (..., 3)
            torch.zeros_like(x)
        )

        # Hessian
        d2e_dz2 = e * (2 * inv**3 - inv**4)  # (..., 1)
        dz_dx_i = dz_dx.unsqueeze(-1)  # (..., 3, 1)
        dz_dx_j = dz_dx.unsqueeze(-2)  # (..., 1, 3)
        outer = dz_dx_i * dz_dx_j  # (..., 3, 3)

        d2z_dx2 = 2 * torch.eye(3, device=x.device) / r_sq  # (3, 3)
        hess_inside = (
            d2e_dz2[..., None, None] * outer +
            de_dz[..., None, None] * d2z_dx2
        )  # (..., 3, 3)

        hess = torch.where(
            inside.unsqueeze(-1),
            hess_inside,
            torch.zeros_like(hess_inside)
        )

        return epsilon.item(), grad, hess


    def query_distance(self, x, distance_type = None, radius=0.1, R_robot=None, S_robot=None, epsilon=0.):
        # Queries varieties of distance from x to the GSplat.
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if distance_type == 'ball-to-ball':
            ball_radius = torch.max(self.scales, dim=-1)[0]
            dist, grad, hess = batch_point_distance(x[..., :3].squeeze(), self.means)

            h = dist - (ball_radius + radius + epsilon)
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == 'ball-to-ball-squared': 
            ball_radius = torch.max(self.scales, dim=-1)[0]
            squared_dist, grad, hess = batch_squared_point_distance(x[..., :3].squeeze(), self.means)

            h = squared_dist - (ball_radius + radius + epsilon)**2
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == 'ball-to-pt-squared': 
            squared_dist, grad, hess = batch_squared_point_distance(x[..., :3].squeeze(), self.means)

            h = squared_dist - (radius + epsilon)**2
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == 'mahalanobis':
            maha_dist, grad, hess = batch_mahalanobis_distance(x[..., :3].squeeze(), self.means, self.covs_inv)

            h = maha_dist - 1.
            grad_h = grad
            hess_h = hess

            info = None

        elif (distance_type is None) or (distance_type == 'ball-to-ellipsoid'):
            # Queries the min Euclidian distance from point to ellipsoid
            # Rotate point into the ellipsoid frame

            # Convert rotations from quaternions to rotation matrices
            # rots = quaternion_to_rotation_matrix(self.rots)
            rots = self.rots
            # rots = torch.matmul(self.Rnerf_world, torch.matmul(rots, self.Rnerf_world.T))
            # print(rots.shape, "good job2")

            # Sort the scales in descending order as required by the solver
            sorted_output = torch.sort(self.scales, dim=-1, descending=True)
            sorted_scales, sorted_inds = sorted_output[0], sorted_output[1]
      
            # NOTE:!!! IMPORTANT!!! When we sort, we need to change the rotation matrices accordingly
            rots = torch.gather(rots, 2, sorted_inds[..., None, :].expand_as(rots))

            # Translate robot w.r.t ellipsoid mean, then rotate point into ellipsoid aligned frame
            x_local_frame = torch.bmm( torch.transpose(rots, 1, 2) , (x[..., :3] - self.means).unsqueeze(-1) ).squeeze() + 1e-8

            # The solver requires the point to be in the first octant. Calculate the sign of the point and flip the point.
            flip = torch.sign(x_local_frame)
            x_local_frame = torch.abs(x_local_frame)

            # solve for the distance in the local frame
            dist, _, hess, yhat = distance_point_ellipsoid(sorted_scales + 1e-8, x_local_frame)

            # flip, rotate, and translate the closest point back to the global frame
            y = torch.bmm(rots, (flip * yhat).unsqueeze(-1)).squeeze(-1) + self.means

            # Calculate cbf 
            phi = torch.sign( torch.sum( (1./ sorted_scales)**2 * (x_local_frame**2) , dim=-1) - 1.)

            epsilon, grad_eps, hess_eps = self.smooth_bump_epsilon(x[:3])
            # print("x:", x[:3])
            # print("eps:", epsilon)
            h = phi * dist - (radius + epsilon)**2 #update cbf here, to compute gradients for pose uncertainty aswell, maybe hessian aswell.

            # Compute gradient in world frame. 
            grad_h = 2 * phi[..., None] * (x[..., :3] - y) -  2 * (radius + epsilon) * grad_eps

            # Mutliple Hessian by phi
            outer_grad_eps = torch.einsum("bi,bj->bij", grad_eps, grad_eps)
            hess_h = phi[..., None, None] * hess - 2 * outer_grad_eps - 2 * (radius + epsilon) * hess_eps.squeeze(0)

            info = {'y': y, 'phi': phi}

        else:
            raise ValueError('Distance type not recognized. Please provide a valid distance type.')

        return h, grad_h, hess_h, info
