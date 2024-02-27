# State Estimators and NMPC Algorithms in Python

This is an implementation of three state estimator approaches using Casadi library:

- Extended Kalman Filter (EKF)
- Constrained Extended Kalman Filter (CEKF)
- Moving Horizon Estimator (MHE)

This is also an implemantation of three NMPC algorithms:

- Single Shooting
- Multiple Shooting
- Orthogonal Collocation on Finite Elements

These state estimators and NMPC approaches were tested for the Van de Vuse reactor. 

Their performance was presented in an article published in Processes entitled "Influence of Estimators and Numerical Approaches on the Implementation of NMPCs" (https://doi.org/10.3390/pr11041102).

# Codes

The folder of codes contains:

- a code of the implementation of the methods
- two codes of the model, one to be used as internal model of the NMPC and the other as the process
- a code for the application of the methods