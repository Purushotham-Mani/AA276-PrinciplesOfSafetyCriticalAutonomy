from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.training.utils import current_git_hash

torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 64
controller_period = 0.05

start_x = torch.tensor(
    [
        [0, 0, 0, 1, 0.0, 0, 0, 5, 0, 0, 0, 0, 0],
    ]
)
simulation_dt = 0.01

parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
args.gpus = 1

# Define the scenarios
nominal_params = {'g': 9.8}
scenarios = [nominal_params]

# create system dynamics
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.part1 import state_limits
    from solutions.part1 import control_limits
    from solutions.part1 import safe_mask
    from solutions.part1 import failure_mask
    from solutions.part1 import f
    from solutions.part1 import g
else:
    from part1 import state_limits
    from part1 import control_limits
    from part1 import safe_mask
    from part1 import failure_mask
    from part1 import f
    from part1 import g
from neural_clbf.systems import Quad13D
dynamics_model = Quad13D(
    state_limits,
    control_limits,
    safe_mask,
    failure_mask,
    f,
    g,
    nominal_params,
    dt=simulation_dt,
    controller_dt=controller_period,
)

# Initialize the DataModule
initial_conditions = [
    (-3.0 , 3.0 ),
    (-3.0 , 3.0 ),
    (-3.0 , 3.0 ),
    (-1.0 , 1.0 ),
    (-1.0 , 1.0 ),
    (-1.0 , 1.0 ),
    (-1.0 , 1.0 ),
    (-5.0 , 5.0 ),
    (-5.0 , 5.0 ),
    (-5.0 , 5.0 ),
    (-5.0 , 5.0 ),
    (-5.0 , 5.0 ),
    (-5.0 , 5.0 )
]
data_module = EpisodicDataModule(
    dynamics_model,
    initial_conditions,
    trajectories_per_episode=0,
    trajectory_length=1,
    fixed_samples=100000,
    max_points=300000000,
    val_split=0.01,
    batch_size=1024,
    # quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
)

experiment_suite = ExperimentSuite([])

# Initialize the controller
cbf_controller = NeuralCBFController(
    dynamics_model,
    scenarios,
    data_module,
    experiment_suite=experiment_suite,
    cbf_hidden_layers=3,
    cbf_hidden_size=512,
    cbf_lambda=0.3,
    cbf_relaxation_penalty=1e3,
    controller_period=controller_period,
    primal_learning_rate=1e-4,
    scale_parameter=1.0, 
    learn_shape_epochs=1,
    use_relu=True,
    disable_gurobi=True,
)

# Initialize the logger and trainer
tb_logger = pl_loggers.TensorBoardLogger(
    'outputs',
    name='',
)
trainer = pl.Trainer.from_argparse_args(
    args,
    logger=tb_logger,
    reload_dataloaders_every_epoch=True,
    max_epochs=51,
)

# Train
torch.autograd.set_detect_anomaly(True)
trainer.fit(cbf_controller)