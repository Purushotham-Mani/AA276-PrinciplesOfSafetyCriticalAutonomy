import torch
import numpy as np
from problem4_helper import NeuralVF, NeuralCBF


num_samples = 1000
state_dim = 13
sampling_bounds = [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0),
                   (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                   (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
                   (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

samples_np = np.random.uniform(
    low=[b[0] for b in sampling_bounds],
    high=[b[1] for b in sampling_bounds],
    size=(num_samples, state_dim)
)
samples = torch.tensor(samples_np, dtype=torch.float32)

neuralvf = NeuralVF()
neuralcbf = NeuralCBF()

vf_values = neuralvf.values(samples)
safe_mask_vf = vf_values >= 0
safe_volume_vf = safe_mask_vf.sum().item() / num_samples

cbf_values = neuralcbf.values(samples)
safe_mask_cbf = cbf_values >= 0
safe_volume_cbf = safe_mask_cbf.sum().item() / num_samples

print(f"Estimated safe set volume (VF model):  {safe_volume_vf:.4f}")
print(f"Estimated safe set volume (CBF model): {safe_volume_cbf:.4f}")
print(f"Relative difference: {((safe_volume_vf - safe_volume_cbf) / safe_volume_cbf) * 100:.2f}%")
