from Casadi_NMPC import *
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
import numpy as np

"""
If use_EKF and use_CEKF are False, the code will use the MHE estimator.
"""
use_EKF = False
use_CEKF = True


"""
If use_SS and use_MS are False, the code will use the Orthogonal Collocation algorithm for the controller.
"""
use_SS = False
use_MS = False

from VDV4x2_new_plant import *

plant = ODEModel(dt=dt, x=X, u=U, dx=dx,
                 J=J, d=D, p=PP, dm=DM)
plant.get_equations(intg='idas')

# Model

from VDV4x2_new import *

modelo = ODEModel(dt=dt, x=X, u=U, dx=dx,
                  J=J, d=D, p=P, dm=DM, y=Y)
modelo.get_equations(intg='idas')

## Optimizer

opts = {
    'warn_initial_bounds': False, 'print_time': False,
    'ipopt': {'print_level': 1}
}

## Initial Conditions

Caguess = 1.8
Cbguess = 1.1
Trguess = 144.2
Tkguess = 150
Fguess = 100
Tkguess = 150
Qguess = - 0.01  # Q = Q/ Kw*AR = Q / 866.88

xguess = [Caguess, Cbguess, Trguess, Tkguess]
uguess = [Fguess, Qguess]

## Boundaries - Condition of states

lbCa = 0  # 0.1
ubCa = 4
lbCb = 0  # 0.1
ubCb = 3.0
lbTr = 50
ubTr = 200
lbTk = 50
ubTk = 200

lbx = [lbCa, lbCb, lbTr, lbTk]
ubx = [ubCa, ubCb, ubTr, ubTk]

## Boundaries - Condition of manipulated variables

lbF = 10
upF = 400
lbQ = - 10
upQ = 0

lbu = [lbF, lbQ]
ubu = [upF, upQ]

## Boundaries - System disturbance conditions

lbCain = 0.1
ubCain = 6.1
lbTin = 50
ubTin = 200

lbd = [lbCain, lbTin]
upd = [ubCain, ubTin]

## Boundaries - Parameter Conditions

lbCp = 0.5
ubCp = 1.5

lbK01 = 0.5
ubK01 = 1.5

# Create the vector of constraints for CEKF

b = np.empty((modelo.x.shape[0] + modelo.theta.shape[0], 2))

b[0, :] = [lbCa, ubCa]
b[1, :] = [lbCb, ubCb]
b[2, :] = [lbTr, ubTr]
b[3, :] = [lbTk, ubTk]
b[4, :] = [lbCain, ubCain]
b[5, :] = [lbK01, ubK01]
b[6, :] = [lbCp, ubCp]

## Extended Kalman Filter

P0 = np.diag([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3])
Q = np.diag([3e-2, 5e-2, 1e1, 1e1, 1e1, 1e2, 1e2]) * 1e-4
R = np.diag([3e-2, 5e-3, 8e-1, 8e-1])

# Equality constraint matrix is identity

xmin = np.array([lbCa, lbCb, lbTr, lbTk, lbCain, lbK01, lbCp]).reshape(modelo.x.shape[0] + modelo.theta.shape[0], )
xmax = np.array([ubCa, ubCb, ubTr, ubTk, ubCain, ubK01, ubCp]).reshape(modelo.x.shape[0] + modelo.theta.shape[0], )

ymin = np.array([lbCa, lbCb, lbTr, lbTk]).reshape(modelo.x.shape[0], )
ymax = np.array([ubCa, ubCb, ubTr, ubTk]).reshape(modelo.x.shape[0], )

if use_EKF:
    cekf = EKF(dt=dt, P0=P0, Q=Q, R=R, x=X, u=U, y=Y, dx=dx, theta=modelo.theta,
               dm=modelo.dm)

elif use_CEKF:
    cekf = CEKF(dt=dt, P0=P0, Q=Q, R=R, x=X, u=U, y=Y, dx=dx, theta=modelo.theta,
                dm=modelo.dm, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, constrained=True)

else:
    horizon = 3
    Cainguess = 5.1
    K01guess = 1.
    Cpguess = 1.

    lbx_mhe = [lbCa, lbCb, lbTr, lbTk, lbCain, lbK01, lbCp]
    ubx_mhe = [ubCa, ubCb, ubTr, ubTk, ubCain, ubK01, ubCp]
    xguess_mhe = [Caguess, Cbguess, Trguess, Tkguess, Cainguess, K01guess, Cpguess]
    cekf = MHE(dt=dt, P0=P0, Q=Q, R=R, x=X, u=U, y=Y, dx=dx, theta=modelo.theta,
               dm=modelo.dm, N=horizon, xguess=xguess_mhe, lbx=lbx_mhe, ubx=ubx_mhe)

## NMPC

N = 40
M = 10
Q = np.diag([5., 5e-2])
W = np.diag([1e-5, 9e-3])
lbdf = -50
ubdf = 50
lbdQ = -0.15
ubdQ = 0.15
lbdu = [lbdf, lbdQ]
ubdu = [ubdf, ubdQ]

if use_SS:
    nmpc = NMPC_SS(dt=dt, N=N, M=M, Q=Q, W=W, x=X, u=U, c=C, dx=dx,
                   xguess=xguess, uguess=uguess, theta=modelo.theta, dm=modelo.dm, lbx=lbx, ubx=ubx, lbu=lbu,
                   ubu=ubu, lbdu=lbdu, ubdu=ubdu, opts={})

elif use_MS:
    nmpc = NMPC_MS(dt=dt, N=N, M=M, Q=Q, W=W, x=X, u=U, c=C, dx=dx,
                   xguess=xguess, uguess=uguess, theta=modelo.theta, dm=modelo.dm, lbx=lbx, ubx=ubx, lbu=lbu,
                   ubu=ubu, lbdu=lbdu, ubdu=ubdu, opts={})

else:
    nmpc = NMPC_CO(dt=dt, N=N, M=M, Q=Q, W=W, x=X, u=U, c=C, dx=dx,
                   xguess=xguess, uguess=uguess, theta=modelo.theta, dm=modelo.dm, lbx=lbx, ubx=ubx, lbu=lbu,
                   ubu=ubu, lbdu=lbdu, ubdu=ubdu, opts={})

# Initializing
t = 1  # counting
tsim = 2  # hours
nopt = 10  # Time for each optimization round
niter = math.ceil(tsim / (dt * nopt))

xf = [2.4, 1.1, 140, 140]
uf = [85, -0.04]
dist = [5.1, 1, 1]
dm = [130]
xhat = copy.deepcopy(xf)
dhat = copy.deepcopy(dist)
dmhat = copy.deepcopy(dm)

ysim = np.zeros([niter * nopt, 4])
usim = np.zeros([niter * nopt, 2])
spsim = np.zeros([niter * nopt, 2])
dsim = np.zeros([niter * nopt, 3])
xest = np.zeros([niter * nopt, 4])
dest = np.zeros([niter * nopt, 3])
yplanta = np.zeros([niter * nopt, 4])
yplanta_cekf = np.zeros([niter * nopt, 4])

dm_array = []

cpu_time = []

## Simulation

start_time = time.time()

for ksim in range(0, niter):
    start = time.time()  # comp time
    n = ksim / niter
    opts = {
        'warn_initial_bounds': False, 'print_time': False,
        'ipopt': {'print_level': 1, 'constr_viol_tol': 1e-10,
                  'tol': 1e-10}
    }

    if n <= 2 / 4:
        sp = np.array([1.1, 140]).reshape(2, 1)
    else:
        sp = np.array([1.1, 140]).reshape(2, 1)

    for i in range(0, nopt):
        # NMPC
        ctrl = nmpc.calc_control_actions(ksim=ksim + 1, x0=xhat, u0=uf, sp=sp,
                                         theta0=dhat, dm0=dmhat)
        uf = ctrl['U']

        # Disturbances
        if n > 1 / 4 and n < 2 / 4:
            dist = [5.1, K01, Cp]
        elif n >= 2 / 4:
            dist = [6, K01, Cp]
        else:
            dist = [5.1, K01, Cp]

        if n == 3 / 4:
            dm = [130]
            dm_array.append(dm)
        elif n == 1 / 4:
            dm = [130]
            dm_array.append(dm)
        else:
            dm = [130]
            dm_array.append(dm)

        # Plant simulations
        sim = plant.simulate_step(xf=xf, uf=uf, thetaf=dist, dmf=dm)
        ymeas = sim['x'] * (1 + 0.001 * np.random.normal(0, 1))
        xf = sim['x'].ravel()
        ysim[t - 1, :] = xf
        usim[t - 1, :] = sim['u']
        dsim[t - 1, :] = sim['theta']
        spsim[t - 1, :] = np.reshape(sp, [1, len(sp)])
        yplanta[t - 1, :] = ymeas

        # State estimators
        if use_EKF:
            estim = cekf.update_state(xkhat=vertcat(xhat, dhat), uf=uf, dmf=dmhat, ymeas=ymeas)

        elif use_CEKF:
            estim = cekf.update_state(xkhat=vertcat(xhat, dhat), uf=uf, dmf=dmhat, ymeas=ymeas)
            yplanta_cekf[t - 1, :] = estim['ycekf']

        else:
            umhe = ctrl['u'][:horizon - 1, :]
            umhe = np.hstack([umhe[:, 0], umhe[:, 1]])
            estim = cekf.update_state(xkhat=vertcat(xhat, dhat), uf=uf, dmf=dmhat, ymeas=ymeas, umhe=umhe)


        xhat = estim['x']
        dhat = estim['theta']
        xest[t - 1, :] = np.reshape(xhat, [1, len(xhat)])
        dest[t - 1, :] = np.reshape(dhat, [1, len(dhat)])
        dmhat = dm

        t += 1
    end = time.time()
    cpu_time += [end - start]

exec_time = time.time() - start_time

print(f'Tempo de execução = {exec_time} s')

avg_time = np.mean(cpu_time)  # avg time spent at each opt cycle

# Plot
time = np.linspace(0, tsim, niter * nopt)

if use_EKF:
    legenda = 'EKF'

elif use_CEKF:
    legenda = 'CEKF'

else:
    legenda = 'MHE'

dm_array = np.array(dm_array)

fig1, ax1 = plt.subplots(3, 3)  # x
fig1.set_size_inches(8, 8)
fig1.subplots_adjust(wspace=0.5)
ax1[0, 0].plot(time, ysim[:, 0], label='Plant', color='brown')  # Ca
ax1[0, 0].plot(time, xest[:, 0], linestyle='dotted', label=legenda, linewidth=3, color='gray')
ax1[0, 0].set_ylabel(r'$C_{A}$ (mol/L)')
ax1[0, 1].plot(time, ysim[:, 1], label='Plant', color='brown')  # Cb
ax1[0, 1].plot(time, spsim[:, 0], linestyle='--', label='SP', color='black')
ax1[0, 1].plot(time, xest[:, 1], linestyle='dotted', label=legenda, linewidth=3, color='gray')
ax1[0, 1].set_ylabel(r'$C_{B}$ (mol/L)')
# ax1[0, 1].set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
ax1[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fontsize=12)
ax1[0, 2].plot(time, ysim[:, 2], label='Plant', color='brown')  # T
ax1[0, 2].plot(time, spsim[:, 1], linestyle='--', label='SP', color='black')
ax1[0, 2].plot(time, xest[:, 2], linestyle='dotted', label=legenda, linewidth=3, color='gray')
ax1[0, 2].set_ylabel(r'$T_{r}$ (°C)')
ax1[0, 2].set_yticks([138, 139, 140, 141, 142])


ax1[1, 0].plot(time, ysim[:, 3], label='Plant', color='brown')  # T
ax1[1, 0].plot(time, xest[:, 3], linestyle='dotted', label=legenda, linewidth=3, color='gray')
ax1[1, 0].set_ylabel(r'$T_{k}$ (°C)')
ax1[1, 1].step(time, usim[:, 0], color='brown')  # f
ax1[1, 1].set_ylabel(r'F/V ($h^{-1}$)')
ax1[1, 2].step(time, usim[:, 1], color='brown')  # Tk
ax1[1, 2].set_ylabel(r'$Q/(k_{w}.A_{r})$ (K^{-1})')

ax1[2, 0].plot(time, dsim[:, 0], label='Plant', color='brown')  # Ca
ax1[2, 0].plot(time, dest[:, 0], linestyle='dotted', label=legenda, linewidth=3, color='gray')
ax1[2, 0].set_ylabel(r'$C_{Ain}$ (mol/L)')
ax1[2, 0].set_xlabel('t (h)')
ax1[2, 1].plot(time, dsim[:, 1], label='Plant', color='brown')  # Cb
ax1[2, 1].plot(time, dest[:, 1] * K01, linestyle='dotted', label=legenda, linewidth=3, color='gray')
ax1[2, 1].set_ylabel(r'$K01$ ($h^{-1}$)')
ax1[2, 1].set_xlabel('t (h)')
ax1[2, 2].plot(time, dsim[:, 2], label='Plant', color='brown')  # T
ax1[2, 2].plot(time, dest[:, 2] * Cp, linestyle='dotted', label=legenda, linewidth=3, color='gray')
ax1[2, 2].set_ylabel(r'$C_{P}$ (kJ/kg.K)')
ax1[2, 2].set_xlabel('t (h)')

plt.show()

