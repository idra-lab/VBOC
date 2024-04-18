# Learning the Viability Kernel of Robotic Manipulators

This GitHub repository contains data-driven algorithms for the computation of viability kernels of robotic manipulators. 
The algorithms use Optimal Control Problems (OCPs) to generate training data and employ Neural Networks (NNs) to approximate the viability kernel based on this data.

## Algorithms

### Viability-Boundary Optimal Control (VBOC)

The Viability-Boundary Optimal Control algorithm, referred to as "VBOC," utilizes OCPs to directly compute states that lie exactly on the boundary of the viability set and uses an NN regressor to approximate the set.

### Hamilton-Jacoby Reachability (HJR)

The Hamilton-Jacoby Reachability algorithm, referred to as "HJB," is an adaptation of a reachability algorithm presented in the paper "Recursive Regression with Neural Networks: Approximating the HJI PDE Solution" by V. Rubies-Royo and C. Tomlin. 
HJR computes the solution of the Hamilton-Jacoby-Isaacs (HJI) Partial Differential Equation (PDE) through recursive regression. NN classifiers are employed to approximate the set.

### Active Learning (AL)

The Active Learning algorithm, referred to as "AL" solves OCPs for system's initial states to verify from which of these states it is possible to stop (reach the zero-velocity set). 
AL leverages then Active Learning techniques to iteratively select batches of new states to be tested to maximize the resulting NN classifier accuracy.

## Installation
- Clone the repository\
`git clone https://github.com/idra-lab/VBOC.git`
- Install the requirements\
`pip install -r requirements.txt`
- Follow the instructions to install [CasADi](https://web.casadi.org/get/), [Acados](https://docs.acados.org/installation/index.html) and [Pytorch](https://pytorch.org/get-started/locally/).

# Usage
Run the script `main.py` inside the `scripts` folder. One can consult the help for the available options:
```
cd scripts
python3 main.py --help
```
For example:
- find the states on the boundary of the viability kernel through VBOC
```
python3 main.py -v
```
- use NN regression to learn an approximation of the N-control invariant set
```
python3 main.py -t
```

## References

- A. La Rocca, M. Saveriano, A. Del Prete, "VBOC: Learning the Viability Boundary of a Robot Manipulator using Optimal Control", IEEE Robotics and Automation Letters, 2023 
- V. Rubies-Royo and C. Tomlin, "Recursive Regression with Neural Networks: Approximating the HJI PDE Solution", in 5th International Conference on Learning Representations, 2017
- A. Chakrabarty, C. Danielson, S. D. Cairano, and A. Raghunathan, "Active Learning for Estimating Reachable Sets for Systems With Unknown Dynamics", IEEE Transactions on Cybernetics, 2020


