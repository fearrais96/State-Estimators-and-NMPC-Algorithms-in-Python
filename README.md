# State Estimators and NMPC Algorithms in Python
> The aim of this project was develop a Python package containing nonlinear model predictive control (NMPC) and state estimator algorithms. This package was succesfully applied for the Van de Vusse reactor benchmark while considering set-point changes, unreachable set-point, disturbances, and mismatches. The

## 📖 Overview
This repository contains the source code for State Estimators and NMPC Algorithms in Python, developed by Fernando Arrais Romero Dias Lima, Ruan de Rezende Faria , Rodrigo Curvelo , Matheus Calheiros Fernandes Cadorini, César Augusto García Echeverry, Maurício Bezerra de Souza Jr. and Argimiro Resende Secchi at the Laboratory of Software Development for Process Control and Optimization (LADES) - COPPE/UFRJ, in association with the Federal University of Rio de Janeiro.

The purpose of this project is to develop a package to use state estimators and nonlinear model predictive controllers (NMPC) in Python. This repository serves as a resource for research, development, and collaboration within the field of process control.

## 🚀 Features
This is an implementation of three state estimator approaches using Casadi library:

- Extended Kalman Filter (EKF)
- Constrained Extended Kalman Filter (CEKF)
- Moving Horizon Estimator (MHE)

This is also an implemantation of three NMPC algorithms:

- Single Shooting
- Multiple Shooting
- Orthogonal Collocation on Finite Elements

## 📦 Installation
To install and use this project, follow these steps:

### Prerequisites
Ensure you have the following dependencies installed:
```bash
# Example for Python projects
pip install casadi
pip install numpy
pip install qpsolvers[open_source_solvers]
```

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/LADES-PEQ/State-Estimators-and-NMPC-Algorithms-in-Python.git
   ```
2. Navigate to the project directory:
   ```bash
   cd State-Estimators-and-NMPC-Algorithms-in-Python
   cd Codes
   ```
3. Run the main script:
   ```bash
   Casadi_NMPC.py
   ```
4. (Optional) Configure environment variables or additional settings.

## 📂 Repository Structure
```
├── Codes                       # Folder with Codes
   ├── Casadi_NMPC.py           # Implemenation of the algorithms
   ├── VDV4x2_new.py            # NMPC model
   ├── VDV4x2_new_plant.py      # Process
   └── VDV_NMPC_new_4x2.py      # Apply the algorithms
├── LICENSE                     # License information
└── README.md                   # Project documentation
```

## ✏️ Authors & Contributors
This project was developed by **Laboratory of Software Development for Process Control and Optimization (LADES) - COPPE/UFRJ** under the coordination of **Argimiro Resende Secchi**.

- **Fernando Arrais Romero Dias Lima** - Code development and paper writing - farrais@eq.ufrj.br
- **Ruan de Rezende Faria** - Code development and paper writing
- **Rodrigo Curvelo** - Code development
- **Matheus Calheiros Fernandes Cadorini** - Paper writing
- **César Augusto García Echeverry** - Paper writing
- **Maurício Bezerra de Souza Jr.** - Supervision
- **Argimiro Resende Secchi** - Supervision

We welcome contributions!

## 🔬 References & Publications
If you use this work in your research, please cite the following publications:
- **Fernando Arrais Romero Dias Lima, Ruan de Rezende Faria , Rodrigo Curvelo , Matheus Calheiros Fernandes Cadorini, César Augusto García Echeverry, Maurício Bezerra de Souza Jr. and Argimiro Resende Secchi. "Influence of Estimators and Numerical Approaches on the Implementation of NMPCs." Processes 2023, 11(4), 1102, https://doi.org/10.3390/pr11041102 .**
- **GitHub Repository**: https://github.com/LADES-PEQ/State-Estimators-and-NMPC-Algorithms-in-Python.git

BibTeX:
```bibtex
@article{Lima2023processes,
title={Influence of Estimators and Numerical Approaches on the Implementation of NMPCs},
author={Lima, Fernando Arrais Romero Dias and Faria, Ruan de Rezende and Curvelo, Rodrigo and Cadorini, Matheus Calheiros Fernandes and Echeverry, C{\'e}sar Augusto Garc{\'\i}a and de Souza Jr, Maur{\'\i}cio Bezerra and Secchi, Argimiro Resende},
journal={Processes},
volume={11},
number={4},
pages={1102},
year={2023},
DOI = {https://doi.org/10.3390/pr11041102}
}
```

## 🛡 License
This work is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0) License**.  
You are free to:
- **Use, modify, and distribute** this code for any purpose.
- **Cite the following reference** when using this code:

  **Fernando Arrais Romero Dias Lima, Ruan de Rezende Faria , Rodrigo Curvelo , Matheus Calheiros Fernandes Cadorini, César Augusto García Echeverry, Maurício Bezerra de Souza Jr. and Argimiro Resende Secchi**.  
  **"Influence of Estimators and Numerical Approaches on the Implementation of NMPCs"**,  
  *Processes*, vol. 11, no. 4, 2023.  
  [DOI: https://doi.org/10.3390/pr11041102]

See the full license details in the LICENSE.txt.

## 📞 Contact
For any inquiries, please contact **Fernando Arrais Romero Dias Lima** at **farrais@eq.ufrj.br** or open an issue in this repository.

## ⭐ Acknowledgments
We acknowledge the support of **Federal University of Rio de Janeiro**, **Capes**, and all contributors.
