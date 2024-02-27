# Model parameters for the Van de Vusse CSTR: 4 states and 2 inputs

from casadi import *

# Constants
K01 = 1.287e12  # frequency factor K01 [h^-1]
K02 = 1.287e12  # frequency factor K02 [h^-1]
K03 = 9.043e9  # frequency factor K03 [L/mol.h]
R_gas = 8.3144621e-3  # universal gas constant
EA1 = 9758.3  # Activation Energy EA1  [kj/mol]
EA2 = 9758.3  # Activation Energy EA2  [kj/mol]
EA3 = 8560.0  # Activation Energy EA3  [kj/mol]
HR1 = 4.2  # Reaction enthalpy HR1   [kj/mol A]
HR2 = -11.0  # Reaction enthalpy HR2   [kj/mol B]
HR3 = -41.85  # Reaction enthalpy HR3   [kj/mol C]
rho = 0.9342  # Density [kg/l]
Cp = 3.01  # Heat capacity [kj/Kg.K]
mkCpk = 10.0  # Heat supplied by the coolant [kj/k]
AR = 0.215  # Reactor wall area [m^2]
VR = 10.01  # Volume of the reactor [l]
Kw = 4032.0  # Heat transfer coefficient [kj/h.m^2.K]
CA0 = 5.1  # Input concentration of component A [mol/l]
mk = 5
cpk = 2

# States of the model
Ca = MX.sym('Ca', 1)  # concentration of A (mol/L)
Cb = MX.sym('Cb', 1)  # concentration of B (mol/L)
Tr = MX.sym('Tr', 1)  # Reactor temperature (°C)
Tk = MX.sym('Tk', 1)  # Jacket temperature (°C)

X = vertcat(Ca, Cb, Tr, Tk)
C = vertcat(Cb, Tr)  # controlled variables

# Model outputs
Y = vertcat(Ca, Cb, Tr, Tk)

# Model inputs
F = MX.sym('F/V', 1)  # space valocity (h^-1)
Q_KwAr = MX.sym('Qk/(Kw*AR)', 1)  # heat supplied by the jacket (kJ/h)
# divided by the reactor wall area [m^2] and the heat transfer coefficient [kj/h.m^2.K]
U = vertcat(F, Q_KwAr)  # Vector containing the inputs

# Disturbances
Cain = MX.sym('Cain', 1)  # Input concentration of component A
D = vertcat(Cain)  # Disturbance vector

# Measured Disturbances
T_in = MX.sym('T_in', 1)  # Input temperature
DM = vertcat(T_in)

K01 = MX.sym('K01', 1)
Cp = MX.sym('Cp', 1)
PP = vertcat(K01, Cp)

# Auxiliary Terms

K1 = K01 * exp((-EA1) / ((Tr + 273.15)))
K2 = K02 * exp((- EA2) / ((Tr + 273.15)))
K3 = K03 * exp((- EA3) / ((Tr + 273.15)))

# ODE System
dCadt = F * (Cain - Ca) - Ca * K1 - K3 * Ca ** 2
dCbdt = - F * Cb + Ca * K1 - Cb * K2
dTrdt = (K1 * Ca * HR1 + K2 * Cb * HR2 + K3 * (Ca ** 2) * HR3) / (- rho * Cp) \
        + F * (T_in - Tr) + (((Kw * AR) * (Tk - Tr)) / (rho * Cp * VR))
dTkdt = (Q_KwAr * Kw * AR + Kw * AR * (Tr - Tk)) / (mk * cpk)

dx = vertcat(dCadt, dCbdt, dTrdt, dTkdt)

# Cost function
J = - (Cb / (Cain - Ca) + (Cb / Cain) - 1.3e-4 * Tk)

# Sampling time
dt = 2.5e-3