# Stanford AA276: Principles of Safety-Critical Autonomy, Spring 2025

## Homework 1

In this homework assignment, you will gain experience working with an off-the-shelf<br>
neural CBF library and training your own neural CBFs for a 13D quadrotor system!

There are **3 parts**, which you must complete by following the instructions in<br>
`./part1.py`, `./part2.py`, and `./part3.py`.

## Deliverables
You should submit the three files: `./part1.py`, `./part2.py`, and `./part3.py`,<br>
as well as the false safety rate reported and plot generated in Part 3.<br>
Also remember to answer and submit the writeup questions.

## Environment Setup

First, you should follow `../README.md`, which is located in the [previous directory](..).<br>
Create your VM on Google Compute Engine following [this tutorial](https://docs.google.com/document/d/1btNdbHUv_ErEz2ccBMA5xL8scMc3octeCv9BheKb0wQ/edit?usp=sharing).

You can maintain two different virtual environments for running the **test scripts** (locally)<br>
and the **GPU-based scripts** (in Google Compute Engine).

To run the **test scripts**, you need to install the packages in `test_requirements.txt`:<br>
`pip install -r test_requirements.txt`

To run the **GPU-based scripts**, you need to install the packages in `gpu_requirements.txt`<br>
as well as `../libraries/neural_clbf` as an editable package. Within the Google Compute Engine<br>
virtual machine, you can do this with:<br>
`sudo chmod +x ./setup.sh`<br>
`./setup.sh`