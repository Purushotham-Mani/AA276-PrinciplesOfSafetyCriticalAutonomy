# Stanford AA276: Principles of Safety-Critical Autonomy, Spring 2025

## Homework 2

In this homework assignment, you will gain experience working with off-the-shelf<br>
grid-based and learning-based HJ reachability libraries to compute your own value functions for a 13D quadrotor system!

## Instructions
Follow the instructions provided in the Homework 2 handout.

## Environment Setup

### Problem 4

Since you will likely encounter package incompatibilities between the neural HJ reachability library and the neural CBF library, we recommend that you use separate virtual environments and scripts.

#### Virtual Environment Management
`python -m venv [name]`<br>
(On Linux): `source [name]/bin/activate`

#### To use the neural HJ reachability library:
Install required packages with `pip install -r requirements.txt`.<br>
Install PyTorch with CUDA for your system: https://pytorch.org/.
For example, for a Linux system with CUDA 12.4:<br>
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

#### To use the neural CBF library:
Follow the instructions previously provided in Homework 1.<br>
We recommend that you write scripts that use the neural CBF library (i.e., when using NeuralCBF from `problem4_helper.py`) in your `../hw1/` folder from Homework 1, since the code needs your solution files from Homework 1.
Alternatively, you can copy your solution files from Homework 1 to this directory.
A copy of `problem4_helper.py` should already be in `../hw1/` for you to use.
If you did not save your `cbf.ckpt` from Homework 1, you can download one that we provide with:<br>
`pip install gdown`<br>
`gdown https://drive.google.com/uc?id=1xN9UX2VYcYZ4ohU5tTjcDas0WlBQG3_n`