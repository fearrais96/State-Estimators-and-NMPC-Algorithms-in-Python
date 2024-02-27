import copy

import numpy as np
from casadi import *
from scipy.linalg import expm
from qpsolvers import solve_qp


class ODEModel:
    """
    This class creates an ODE model using casadi symbolic framework
    """

    def __init__(self, dt, x, dx, J, y=None, u=None, d=None, p=None, dm=None):
        self.dt = dt  # sampling
        self.x = x  # states (sym)
        self.y = MX.sym('y', 0) if y is None else y  # outputs (sym)
        self.u = MX.sym('u', 0) if u is None else u  # inputs (sym)
        self.d = MX.sym('d', 0) if d is None else d  # disturbances (sym)
        self.dm = MX.sym('dm', 0) if dm is None else dm  # measured disturbances (sym)
        self.p = MX.sym('p', 0) if p is None else p  # parameters (sym)
        self.dx = dx  # model equations
        self.J = J  # cost function
        self.theta = vertcat(self.d, self.p)  # parameters to be estimated vector (sym)

    def get_equations(self, intg='idas'):
        """
        Gets equations and integrator
        """

        self.ode = {
            'x': self.x,
            'p': vertcat(self.u, self.theta, self.dm),
            'ode': self.dx, 'quad': self.J
        }  # ODE model
        self.F = Function('F', [self.x, self.u, self.theta, self.dm], [self.dx, self.J, self.y],
                          ['x', 'u', 'theta', 'dm'], ['dx', 'J', 'y'])  # model function
        self.rfsolver = rootfinder('rfsolver', 'newton', self.F)  # rootfinder
        self.opts = {'tf': self.dt}  # sampling time
        self.Plant = integrator('F', intg, self.ode, self.opts)  # integrator

    def steady(self, xguess=None, uf=None, thetaf=None, dmf=None):
        """
        Calculates root
        """
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uf = [] if uf is None else uf
        thetaf = [] if thetaf is None else thetaf
        dmf = [] if dmf is None else dmf
        sol = self.rfsolver(x=xguess, u=uf, theta=thetaf, dm=dmf)
        return {
            'x': sol['dx'].full(),
            'J': sol['J'].full()
        }

    def simulate_step(self, xf, uf=None, thetaf=None, dmf=None):
        """
        Simulates 1 step
        """

        uf = [] if uf is None else uf
        thetaf = [] if thetaf is None else thetaf
        dmf = [] if dmf is None else dmf
        Fk = self.Plant(x0=xf, p=vertcat(uf, thetaf, dmf))  # integration
        return {
            'x': Fk['xf'].full().reshape(-1),
            'u': uf, 'theta': thetaf, 'dm': dmf
        }

    def check_steady(self, nss, t, cov, ysim):
        """
        Steady-state identification
        """

        s2 = [0] * len(cov)
        M = np.mean(ysim[t - nss:t, :], axis=0)
        for i in range(0, nss):
            s2 += np.power(ysim[t - i - 1, :] - M, 2)
        S2 = s2 / (nss - 1)
        if np.all(S2 <= cov):
            flag = True
        else:
            flag = False
        return {
            'Status': flag,
            'S2': S2
        }

    def build_nlp_steady(self, xguess=None, uguess=None, lbx=None, ubx=None,
                         lbu=None, ubu=None, opts={}):
        """
        Builds steady-state optimization NLP
        """

        # Guesses and bounds
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []

        # Start NLP
        self.w += [self.x, self.u]
        self.w0 += list(xguess)
        self.w0 += list(uguess)
        self.lbw += list(lbx)
        self.lbw += list(lbu)
        self.ubw += list(ubx)
        self.ubw += list(ubu)
        self.g += [vertcat(self.dx)]
        self.lbg += list(np.zeros(self.dx.shape[0]))
        self.ubg += list(np.zeros(self.dx.shape[0]))

        nlp = {
            'x': vertcat(*self.w),
            'p': self.theta,
            'f': self.J,
            'g': vertcat(*self.g)
        }

        # Solver
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

    def optimize_steady(self, ksim=None, thetaf=[]):
        """
        Performs 1 optimization step (thetaf must be lists)
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=thetaf,
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step ' + str(ksim) + ': Solver did not converge.')
            else:
                print('Optimization step ' + str(ksim) + ': Optimal Solution Found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step: Solver did not converge.')
            else:
                print('Optimization step: Optimal Solution Found.')

                # Solution
        wopt = sol['x'].full()  # solution
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        return {
            'x': wopt[:self.x.shape[0]],
            'u': wopt[-self.u.shape[0]:]
        }

    def build_nlp_dyn(self, N, M, xguess, uguess, lbx=None, ubx=None, lbu=None,
                      ubu=None, m=3, pol='legendre', opts={}):
        """
        Build dynamic optimization NLP
        """

        self.m = m
        self.N = N
        self.M = M
        self.pol = pol

        # Guesses and bounds
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])

        # Polynomials
        self.tau = np.array([0] + collocation_points(self.m, self.pol))
        self.L = np.zeros((self.m + 1, 1))
        self.Ldot = np.zeros((self.m + 1, self.m + 1))
        self.Lint = self.L
        for i in range(0, self.m + 1):
            coeff = 1
            for j in range(0, self.m + 1):
                if j != i:
                    coeff = np.convolve(coeff, [1, -self.tau[j]]) / \
                            (self.tau[i] - self.tau[j])
            self.L[i] = np.polyval(coeff, 1)
            ldot = np.polyder(coeff)
            for j in range(0, self.m + 1):
                self.Ldot[i, j] = np.polyval(ldot, self.tau[j])
            lint = np.polyint(coeff)
            self.Lint[i] = np.polyval(lint, 1)

        # "Lift" initial conditions
        xk = MX.sym('x0', self.x.shape[0])  # first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        uk_prev = uguess

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # Start NLP
        self.w += [xk]
        self.w0 += list(xguess)
        self.lbw += list(lbx)
        self.ubw = list(ubx)
        self.g += [xk - x0_sym]
        self.lbg += list(np.zeros(self.dx.shape[0]))
        self.ubg += list(np.zeros(self.dx.shape[0]))

        # NLP build
        for k in range(0, self.N):
            xki = []  # state at collocation points
            for i in range(0, self.m):
                xki.append(MX.sym('x_' + str(k + 1) + '_' + str(i + 1), self.x.shape[0]))
                self.w += [xki[i]]
                self.lbw += list(lbx)
                self.ubw += list(ubx)
                self.w0 += list(xguess)

            # uk as decision variable
            uk = MX.sym('u_' + str(k + 1), self.u.shape[0])
            self.w += [uk]
            self.lbw += list(lbu)
            self.ubw += list(ubu)
            self.w0 += list(uguess)

            if k >= self.M:
                self.g += [uk - uk_prev]  # delta_u
                self.lbg += list(np.zeros(self.u.shape[0]))
                self.ubg += list(np.zeros(self.u.shape[0]))

            uk_prev = uk

            # Loop over collocation points
            xk_end = self.L[0] * xk
            for i in range(0, self.m):
                xk_end += self.L[i + 1] * xki[i]  # add contribution to the end state
                xc = self.Ldot[0, i + 1] * xk  # expression for the state derivative at the collocation poin
                for j in range(0, m):
                    xc += self.Ldot[j + 1, i + 1] * xki[j]
                fi = self.F(xki[i], uk, self.theta)  # model and cost function
                self.g += [self.dt * fi[0] - xc]  # model equality contraints reformulated
                self.lbg += list(np.zeros(self.x.shape[0]))
                self.ubg += list(np.zeros(self.x.shape[0]))
                # self.J += self.dt*fi[1]*self.Lint[i+1] #add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            xk = MX.sym('x_' + str(k + 2), self.x.shape[0])
            self.w += [xk]
            self.lbw += list(lbx)
            self.ubw += list(ubx)
            self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk - xk_end]
            self.lbg += list(np.zeros(self.x.shape[0]))
            self.ubg += list(np.zeros(self.x.shape[0]))

        self.J = fi[1]

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym, self.theta)
        }

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def optimize_dyn(self, xf, thetaf=[], ksim=None):
        """
        Performs 1 optimization step
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(xf, thetaf),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step ' + str(ksim) + ': Solver did not converge.')
            else:
                print('Optimization step ' + str(ksim) + ': Optimal Solution Found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step: Solver did not converge.')
            else:
                print('Optimization step: Optimal Solution Found.')

        # Solution
        wopt = sol['x'].full()  # solution
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x.shape[0]))
        uopt = np.zeros((self.N, self.u.shape[0]))
        for i in range(0, self.x.shape[0]):
            xopt[:, i] = wopt[i::self.x.shape[0] + self.u.shape[0] +
                                 self.x.shape[0] * self.m].reshape(-1)  # optimal state
        for i in range(0, self.u.shape[0]):
            uopt[:, i] = wopt[self.x.shape[0] + self.x.shape[0] * self.m + i::self.x.shape[0] +
                                                                              self.x.shape[0] * self.m + self.u.shape[
                                                                                  0]].reshape(-1)  # optimal inputs
        return {
            'x': xopt,
            'u': uopt
        }


class EKF:
    """
    This class creates an Extended Kalman Filter using casadi
    symbolic framework (regular EKF if there's no theta)
    """

    def __init__(self, dt, P0, Q, R, x, u, y, dx, theta, dm):
        self.x = x
        self.u = u
        self.y = y
        self.theta = theta
        self.dm = dm
        self.dt = dt
        self.x_ = vertcat(self.x, self.theta)  # extended state vector
        self.Q = Q  # process noise covariance matrix
        self.R = R  # measurement noise covariance matrix
        self.Pk = copy.deepcopy(P0)  # estimation error covariance matrix

        # Model equations
        dx_ = []
        for i in range(0, self.x.shape[0]):
            dx_.append(x[i] + dt * dx[i])
        for j in range(0, self.theta.shape[0]):
            dx_.append(theta[j])
        self.dx_ = vertcat(*dx_)
        self.F_ = Function('F_EKF', [self.x_, self.u, self.dm], [self.dx_])  # state equation
        self.JacFx_ = Function('JacFx_EKF', [self.x_, self.u, self.dm],
                               [jacobian(self.dx_, self.x_)])  # jacobian of F respective to x
        self.H_ = Function('H_EKF', [self.x_, self.u, self.dm], [self.y])  # output equation
        self.JacHx_ = Function('JacHx_EKF', [self.x_, self.u, self.dm],
                               [jacobian(self.y, self.x_)])  # jacobian of H respective to x

    def update_state(self, xkhat, uf, dmf, ymeas):
        """
        Performs 1 model update step
        """

        Fk = self.JacFx_(xkhat, uf, dmf).full()
        xkhat_pri = self.F_(xkhat, uf, dmf).full()  # priori estimate of xk
        Pk_pri = Fk @ self.Pk @ Fk.transpose() + self.Q  # priori estimate of Pk
        Hk = self.JacHx_(xkhat_pri, uf, dmf).full()
        Kk = (Pk_pri @ Hk.T) @ (np.linalg.inv(Hk @ Pk_pri @ Hk.T + self.R))  # Kalman gain
        xkhat_pos = xkhat_pri + Kk @ ((ymeas - self.H_(xkhat_pri, uf, dmf)).full())  # posteriori estimate of xk
        self.Pk_pos = (np.eye(self.x.shape[0] + self.theta.shape[0]) - Kk @ Hk) @ Pk_pri  # posteriori estimate of Pk
        self.Pk = copy.deepcopy(self.Pk_pos)

        # Estimations
        return {
            'x': xkhat_pos[:self.x.shape[0]],
            'theta': xkhat_pos[-self.theta.shape[0]:]
        }


class CEKF:
    """
    This class creates a Constrained Extended Kalman Filter using casadi
    """

    def __init__(self, dt, P0, Q, R, x, u, y, dx, theta, dm, xmin, xmax, ymin, ymax, constrained):
        self.x = x
        self.u = u
        self.y = y
        self.theta = theta
        self.dm = dm
        self.dt = dt
        self.x_ = vertcat(self.x, self.theta)  # extended state vector
        self.Q = Q  # process noise covariance matrix
        self.R = R  # measurement noise covariance matrix
        self.Pk = copy.deepcopy(P0)  # estimation error covariance matrix
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.constrained = constrained

        # Model equations
        dx_ = []
        for i in range(0, self.x.shape[0]):
            dx_.append(x[i] + dt * dx[i])
        for j in range(0, self.theta.shape[0]):
            dx_.append(theta[j])
        self.dx_ = vertcat(*dx_)
        self.F_ = Function('F_EKF', [self.x_, self.u, self.dm], [self.dx_])  # state equation
        self.JacFx_ = Function('JacFx_EKF', [self.x_, self.u, self.dm],
                               [jacobian(self.dx_, self.x_)])  # jacobian of F respective to x
        self.H_ = Function('H_EKF', [self.x_, self.u, self.dm], [self.y])  # output equation
        self.JacHx_ = Function('JacHx_EKF', [self.x_, self.u, self.dm],
                               [jacobian(self.y, self.x_)])  # jacobian of H respective to x

    def update_state(self, xkhat, uf, dmf, ymeas):
        """
        Performs 1 model update step
        """
        Fk = self.JacFx_(xkhat, uf, dmf).full()
        xkhat_pri = self.F_(xkhat, uf, dmf).full()  # priori estimate of xk
        Pk_pri = Fk @ self.Pk @ Fk.transpose() + self.Q  # priori estimate of Pk
        Hk = self.JacHx_(xkhat_pri, uf, dmf).full()
        Kk = (Pk_pri @ Hk.T) @ (np.linalg.inv(Hk @ Pk_pri @ Hk.T + self.R))  # Kalman gain
        xkhat_pos = xkhat_pri + Kk @ ((ymeas - self.H_(xkhat_pri, uf, dmf)).full())  # posteriori estimate of xk
        self.Pk_pos = (np.eye(self.x.shape[0] + self.theta.shape[0]) - Kk @ Hk) @ Pk_pri  # posteriori estimate of Pk

        if self.constrained:

            self.phi = expm(Fk * self.dt)

            # Cálculo da matriz de covariância
            Pk_pri = (self.phi @ self.Pk_pos @ self.phi.T) - ((self.phi @ self.Pk_pos @ Hk.T) @
                                                              np.linalg.inv(Hk @ self.Pk_pos @ Hk.T + self.R) @ (
                                                                          Hk @ self.Pk_pos @ self.phi.T)) + self.Q  # cálculo discreto

            # Cálculo da matriz de ganho
            Kk = (Pk_pri @ Hk.T) @ (np.linalg.inv(Hk @ Pk_pri @ Hk.T + self.R))
            self.Pk_pos = Pk_pri - (Kk @ Hk @ Pk_pri)

            # Matriz Sk
            Sk = np.zeros((len(self.Pk_pos) + len(self.R), len(self.Pk_pos) + len(self.R)))

            Sk[:len(self.Pk_pos), :len(self.Pk_pos)] = np.linalg.inv(self.Pk_pos)
            Sk[:len(self.Pk_pos), len(self.Pk_pos): len(Sk)] = np.zeros((len(self.Pk_pos), len(self.R)))
            Sk[len(self.Pk_pos): len(Sk), :len(self.Pk_pos)] = np.zeros((len(self.R), len(self.Pk_pos)))
            Sk[len(self.Pk_pos): len(Sk), len(self.Pk_pos): len(Sk)] = np.linalg.inv(self.R)

            Sk = (Sk + Sk.T) / 2
            self.Sk = Sk
            # Vetor d
            dk = np.zeros((len(Sk))).reshape(len(Sk), )

            # Restrições de igualdade
            Aeq = np.zeros((len(Hk), len(Sk)))
            Aeq[:Hk.shape[0], :Hk.shape[-1]] = Hk
            Aeq[:Hk.shape[0], Hk.shape[-1]: len(Sk)] = np.eye(len(Hk))
            self.Aeq = Aeq

            Beq = np.array(ymeas - (Hk @ xkhat)).reshape(len(Hk))
            self.Beq = Beq

            # Restrições de desigualdade
            A = np.zeros((len(Sk) * 2, len(Sk)))
            A[:len(Sk), :len(Sk)] = - np.eye(len(Sk))
            A[len(Sk): 2 * len(Sk), :len(Sk)] = np.eye(len(Sk))
            self.A = A

            B = np.zeros((len(Sk) * 2))
            B[:len(xkhat_pos)] = xkhat_pos.reshape(len(xkhat_pos), ) - self.xmin
            B[len(xkhat_pos): len(Sk)] = ymeas.reshape(len(ymeas), ) - self.ymin
            B[len(Sk): len(Sk) + len(xkhat_pos)] = self.xmax - xkhat_pos.reshape(len(xkhat_pos), )
            B[len(Sk) + len(xkhat_pos): 2 * len(Sk)] = self.ymax - ymeas.reshape(len(ymeas), )
            self.B = B

            self.xcon = solve_qp(P=Sk, q=dk, G=A, h=B, A=Aeq, b=Beq, solver="osqp", eps_abs=0.00000001,
                                 eps_rel=0.00000001)

            if id(self.xcon) != id(None):
                xkmais = xkhat_pos.reshape(len(xkhat_pos), ) + self.xcon[:len(xkhat_pos)]
                ymeas = np.array((Hk @ xkhat_pos)).reshape(len(Hk), ) + self.xcon[len(xkhat_pos): len(self.xcon)]
            else:
                xkmais = xkhat_pos
                ymeas = np.array((Hk @ xkhat_pos)).reshape(len(Hk), )

            self.Pk = copy.deepcopy(self.Pk_pos)

        # Estimations
        return {
            'x': xkmais[:self.x.shape[0]],
            'theta': xkmais[-self.theta.shape[0]:],
            'ycekf': ymeas
        }

class MHE:
    """
    This class creates a Moving Horizon Estimator using casadi
    """

    def __init__(self, dt, P0, Q, R, x, u, y, dx, theta, dm, N, xguess, lbx=None, ubx=None, casadi_integrator=False,
                 opts={}):

        self.x = x
        self.u = u
        self.y = y
        self.theta = theta
        self.x_ = vertcat(self.x, self.theta)  # extended state vector
        self.Q = Q  # process noise covariance matrix
        self.R = R  # measurement noise covariance matrix
        self.Pk = copy.deepcopy(P0)  # estimation error covariance matrix
        self.zk = []
        self.dt = dt
        self.dm = dm
        self.N = N
        self.casadi_integrator = casadi_integrator

        # Constraints
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])

        # Model equations AEKF
        dx_ = []
        for i in range(0, self.x.shape[0]):
            dx_.append(x[i] + dt * dx[i])
        for j in range(0, self.theta.shape[0]):
            dx_.append(theta[j])
        self.dx_ = vertcat(*dx_)
        self.F_ = Function('F_EKF', [self.x_, self.u, self.dm], [self.dx_])  # state equation
        self.JacFx_ = Function('JacFx_EKF', [self.x_, self.u, self.dm],
                               [jacobian(self.dx_, self.x_)])  # jacobian of F respective to x
        self.H_ = Function('H_EKF', [self.x_, self.u, self.dm], [self.y])  # output equation
        self.JacHx_ = Function('JacHx_EKF', [self.x_, self.u, self.dm],
                               [jacobian(self.y, self.x_)])  # jacobian of H respective to x

        # Build MHE
        self.x_mhe = self.x_
        self.y_mhe = self.y
        self.u_mhe = self.u
        self.x0 = MX.sym('x0_par', self.x_mhe.shape[0])
        # self.Q_mhe = MX.sym('Q_par', self.Q.shape[0], self.Q.shape[-1])
        self.z_mhe = MX.sym('z0_par', self.y.shape[0])
        # self.xguess = MX.sym('xguess_par', self.x_mhe.shape[0])
        self.H_mhe = Function('H_mhe', [self.x_mhe, self.u_mhe, self.dm], [self.y])  # output equation

        dx_mhe = []
        for i in range(0, self.x.shape[0]):
            dx_mhe.append(dx[i])
        for j in range(0, self.theta.shape[0]):
            dx_mhe.append(theta[j])
        self.dx_mhe = vertcat(*dx_mhe)

        J = (self.z_mhe - self.y_mhe).T @ self.R @ (self.z_mhe - self.y_mhe)

        if self.casadi_integrator:
            dae = {'x': self.x_mhe, 'p': vertcat(self.u_mhe, self.dm, self.z_mhe), 'ode': self.dx_mhe, 'quad': J}
            opts_dae = {'tf': self.dt}
            self.F = integrator('F', 'idas', dae, opts_dae)

        else:
            M = 4  # RK4 steps per interval
            DT = self.dt / M
            f = Function('f', [self.x_mhe, vertcat(self.u_mhe, self.dm, self.z_mhe)], [self.dx_mhe, J])
            X0 = MX.sym('X0', self.x_mhe.shape[0])
            U = MX.sym('U', self.u_mhe.shape[0] + self.dm.shape[0] + self.z_mhe.shape[0])
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT / 2 * k1, U)
                k3, k3_q = f(X + DT / 2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            self.F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

            # "Lift" initial conditions
        xk = MX.sym('x0', self.x_mhe.shape[0])  # first point at each interval

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []

        self.J = (xk - self.x0).T @ self.Q @ (xk - self.x0)

        # NLP
        self.w += [xk]
        self.w0 += list(xguess)
        self.lbw += list(lbx)  # lower boundarie include parameters
        self.ubw = list(ubx)  # upper boundarie include parameters
        self.g += [xk - self.x0]
        self.lbg += list(np.zeros(self.dx_mhe.shape[0]))
        self.ubg += list(np.zeros(self.dx_mhe.shape[0]))

        umhe = MX.sym('umhe_k', self.u_mhe.shape[0] * (N - 1))
        zmhe = MX.sym('zmhe_k', self.y_mhe.shape[0] * (N))

        # self.y_mhe = self.H_(xk, self.u_mhe[0, :], self.dm)

        # NLP build

        for k in range(0, self.N - 1):
            # Loop over collocation points

            fi = self.F(x0=xk, p=vertcat(vertcat(umhe[k], umhe[k + N - 1]), self.dm,
                                         vertcat(zmhe[k], zmhe[k + N], zmhe[k + (2 * N)], zmhe[k + (3 * N)])))

            xk_end = fi['xf']
            self.J += fi['qf']  # add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            xk = MX.sym('x_' + str(k + 1), self.x_mhe.shape[0])

            self.w += [xk]
            self.lbw += list(lbx)
            self.ubw += list(ubx)
            self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk_end - xk]
            self.lbg += list(np.zeros(self.dx_mhe.shape[0]))
            self.ubg += list(np.zeros(self.dx_mhe.shape[0]))

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(umhe, self.dm, self.x0, zmhe)
        }  # nlp construction

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def update_state(self, xkhat, uf, dmf, ymeas, umhe, ksim=None):
        """
        Performs 1 model update step
        """
        self.zk.append(ymeas)
        ymhe = np.zeros((self.N, self.y_mhe.shape[0]))

        if len(self.zk) < self.N:
            ymhe[:len(self.zk), :] = np.array(self.zk)
            ymhe[len(self.zk):, :] = np.array(self.zk[-1])
        else:
            ymhe[:] = np.array(self.zk)
            del (self.zk[0])

        ymhe = np.hstack([ymhe[:, 0], ymhe[:, 1], ymhe[:, 2], ymhe[:, 3]])

        self.ymhe = ymhe

        Fk = self.JacFx_(xkhat, uf, dmf).full()
        xkhat_pri = self.F_(xkhat, uf, dmf).full()  # priori estimate of xk
        Pk_pri = Fk @ self.Pk @ Fk.transpose() + self.Q  # priori estimate of Pk
        Hk = self.JacHx_(xkhat_pri, uf, dmf).full()
        Kk = (Pk_pri @ Hk.T) @ (np.linalg.inv(Hk @ Pk_pri @ Hk.T + self.R))  # Kalman gain
        xkhat_pos = xkhat_pri + Kk @ ((ymeas - self.H_(xkhat_pri, uf, dmf)).full())  # posteriori estimate of xk
        self.Pk_pos = (np.eye(self.x.shape[0] + self.theta.shape[0]) - Kk @ Hk) @ Pk_pri  # posteriori estimate of Pk

        self.xkhat_pos = xkhat_pos
        self.Q = copy.deepcopy(np.linalg.inv(self.Pk_pos))

        """
        Performs 1 optimization step for the NMPC 
        """
        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(umhe, dmf, xkhat_pos, ymhe),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step ' + str(ksim) + ': MHE solver did not converge.')
            else:
                print('Time step ' + str(ksim) + ': MHE optimal solution found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step: MHE solver did not converge.')
            else:
                print('Time step: MHE optimal solution found.')

        # Solution
        wopt = sol['x'].full()
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x_mhe.shape[0]))
        xopt[0, :] = np.array(xkhat_pos).reshape(-1)

        for k in range(self.x_mhe.shape[0]):
            xopt[1:, k] = wopt[k::self.x_mhe.shape[0]].reshape(-1)

        # Estimations
        return {
            'x': xopt[1, :self.x.shape[0]],
            'theta': xopt[1, -self.theta.shape[0]:]
        }


class NMPC_CO:
    """
    This class creates an NMPC using casadi symbolic framework
    """

    def __init__(self, dt, N, M, Q, W, x, u, c, dx, theta, dm, xguess=None,
                 uguess=None, lbx=None, ubx=None, lbu=None, ubu=None, lbdu=None,
                 ubdu=None, m=3, pol='legendre', DRTO=False, opts={}):
        self.dt = dt
        self.dx = dx
        self.x = x
        self.c = c
        self.u = u
        self.theta = theta
        self.m = m
        self.pol = pol
        self.N = N
        self.M = M
        self.dm = dm

        '''
        if tgt: #evaluates tracking target inputs
             self.R = R
        else:
            self.R = np.zeros((self.u.shape[0], self.u.shape[0]))
        '''

        self.Q = Q
        self.W = W

        # Guesses
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        lbdu = -inf * np.ones(self.u.shape[0]) if lbdu is None else lbdu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu
        ubdu = -inf * np.ones(self.u.shape[0]) if ubdu is None else ubdu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in lbdu: lbdu = np.array([-inf if v is None else v for v in lbdu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])
        if None in ubdu: ubdu = np.array([-inf if v is None else v for v in ubdu])

        # Quadratic cost function
        self.sp = MX.sym('SP', self.c.shape[0])
        # self.target = MX.sym('Target', self.u.shape[0])
        self.uprev = MX.sym('u_prev', self.u.shape[0])
        J = (self.c - self.sp).T @ Q @ (self.c - self.sp) + (self.u - self.uprev).T @ W @ (self.u - self.uprev)
        self.F = Function('F', [self.x, self.u, self.theta, self.dm, self.sp,
                                self.uprev], [self.dx, J], ['x', 'u', 'theta', 'dm',
                                                            'sp', 'u_prev'], ['dx', 'J'])  # NMPC model function

        # Polynomials
        self.tau = np.array([0] + collocation_points(self.m, self.pol))
        self.L = np.zeros((self.m + 1, 1))
        self.Ldot = np.zeros((self.m + 1, self.m + 1))
        self.Lint = self.L
        for i in range(0, self.m + 1):
            coeff = 1
            for j in range(0, self.m + 1):
                if j != i:
                    coeff = np.convolve(coeff, [1, -self.tau[j]]) / (self.tau[i] - self.tau[j])
            self.L[i] = np.polyval(coeff, 1)
            ldot = np.polyder(coeff)
            for j in range(0, self.m + 1):
                self.Ldot[i, j] = np.polyval(ldot, self.tau[j])
            lint = np.polyint(coeff)
            self.Lint[i] = np.polyval(lint, 1)

        # "Lift" initial conditions
        xk = MX.sym('x0', self.x.shape[0])  # first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        u0_sym = MX.sym('u0_par', self.u.shape[0])
        uk_prev = u0_sym

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # NLP
        self.w += [xk]
        self.w0 += list(xguess)
        self.lbw += list(lbx)
        self.ubw = list(ubx)
        self.g += [xk - x0_sym]
        self.lbg += list(np.zeros(self.dx.shape[0]))
        self.ubg += list(np.zeros(self.dx.shape[0]))

        # Check if the setpoints and targets are trajectories
        if not DRTO:
            spk = self.sp
            # targetk = self.target
        else:
            spk = MX.sym('SP_k', 2 * (N + 1))
            # targetk = MX.sym('Target_k', 2*N)

        # NLP build
        for k in range(0, self.N):
            # State at collocation points
            xki = []
            for i in range(0, self.m):
                xki.append(MX.sym('x_' + str(k + 1) + '_' + str(i + 1), self.x.shape[0]))
                self.w += [xki[i]]
                self.lbw += [lbx]
                self.ubw += [ubx]
                self.w0 += [xguess]

            # uk as decision variable
            uk = MX.sym('u_' + str(k + 1), self.u.shape[0])
            self.w += [uk]
            self.lbw += list(lbu)
            self.ubw += list(ubu)
            self.w0 += list(uguess)
            self.g += [uk - uk_prev]  # delta_u

            # Control horizon
            if k >= self.M:
                self.lbg += list(np.zeros(self.u.shape[0]))
                self.ubg += list(np.zeros(self.u.shape[0]))
            else:
                self.lbg += list(lbdu)
                self.ubg += list(ubdu)

            # Loop over collocation points
            xk_end = self.L[0] * xk
            for i in range(0, self.m):
                xk_end += self.L[i + 1] * xki[i]  # add contribution to the end state
                xc = self.Ldot[0, i + 1] * xk  # expression for the state derivative at the collocation point
                for j in range(0, m):
                    xc += self.Ldot[j + 1, i + 1] * xki[j]
                if not DRTO:  # check if the setpoints and targets are trajectories
                    fi = self.F(xki[i], uk, self.theta, self.dm, spk, uk_prev)
                else:
                    fi = self.F(xki[i], uk, self.theta, self.dm, vertcat(spk[k], spk[k + N + 1]), uk_prev)
                self.g += [self.dt * fi[0] - xc]  # model equality contraints reformulated
                self.lbg += [np.zeros(self.x.shape[0])]
                self.ubg += [np.zeros(self.x.shape[0])]
                self.J += self.dt * fi[1] * self.Lint[i + 1]  # add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            xk = MX.sym('x_' + str(k + 2), self.x.shape[0])
            self.w += [xk]
            self.lbw += list(lbx)
            self.ubw += list(ubx)
            self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk - xk_end]
            self.lbg += list(np.zeros(self.x.shape[0]))
            self.ubg += list(np.zeros(self.x.shape[0]))

            # u(k-1)
            uk_prev = copy.deepcopy(uk)

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym, u0_sym, self.theta, self.dm, spk)
        }  # nlp construction

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def calc_control_actions(self, x0, u0, sp, theta0=[], dm0=[], ksim=None):
        """
        Performs 1 optimization step for the NMPC
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(x0, u0, theta0, dm0, sp),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step ' + str(ksim) + ': NMPC solver did not converge.')
            else:
                print('Time step ' + str(ksim) + ': NMPC optimal solution found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step: NMPC solver did not converge.')
            else:
                print('Time step: NMPC optimal solution found.')

        # Solution
        wopt = sol['x'].full()
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x.shape[0]))
        uopt = np.zeros((self.N, self.u.shape[0]))
        for i in range(0, self.x.shape[0]):
            xopt[:, i] = wopt[i::self.x.shape[0] + self.u.shape[0] +
                                 self.x.shape[0] * self.m].reshape(-1)  # optimal state
        for i in range(0, self.u.shape[0]):
            uopt[:, i] = wopt[self.x.shape[0] + self.x.shape[0] * self.m + i::self.x.shape[0] +
                                                                              self.x.shape[0] * self.m + self.u.shape[
                                                                                  0]].reshape(-1)  # optimal inputs

        # First control action
        uin = uopt[0, :]
        return {
            'x': xopt,
            'u': uopt,
            'U': uin
        }


class NMPC_MS:
    """
    This class creates an NMPC using casadi symbolic framework
    """

    def __init__(self, dt, N, M, Q, W, x, u, c, dx, theta, dm, xguess=None,
                 uguess=None, lbx=None, ubx=None, lbu=None, ubu=None, lbdu=None,
                 ubdu=None, DRTO=False, opts={}, casadi_integrator=False):
        self.dt = dt
        self.dx = dx
        self.x = x
        self.c = c
        self.u = u
        self.theta = theta
        self.N = N
        self.M = M
        self.dm = dm
        self.casadi_integrator = casadi_integrator

        '''
        if tgt: #evaluates tracking target inputs
             self.R = R
        else:
            self.R = np.zeros((self.u.shape[0], self.u.shape[0]))
        '''

        self.Q = Q
        self.W = W

        # Guesses
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        lbdu = -inf * np.ones(self.u.shape[0]) if lbdu is None else lbdu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu
        ubdu = -inf * np.ones(self.u.shape[0]) if ubdu is None else ubdu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in lbdu: lbdu = np.array([-inf if v is None else v for v in lbdu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])
        if None in ubdu: ubdu = np.array([-inf if v is None else v for v in ubdu])

        # Quadratic cost function
        self.sp = MX.sym('SP', self.c.shape[0])
        # self.target = MX.sym('Target', self.u.shape[0])
        self.uprev = MX.sym('u_prev', self.u.shape[0])
        J = (self.c - self.sp).T @ Q @ (self.c - self.sp) + (self.u - self.uprev).T @ W @ (self.u - self.uprev)

        if casadi_integrator:
            dae = {'x': self.x, 'p': vertcat(self.u, self.theta, self.dm, self.sp, self.uprev), 'ode': self.dx,
                   'quad': J}
            opts_dae = {'tf': self.dt}
            self.F = integrator('F', 'idas', dae, opts_dae)

        else:
            M = 4  # RK4 steps per interval
            DT = self.dt / M
            f = Function('f', [self.x, vertcat(self.u, self.theta, self.dm, self.sp, self.uprev)], [self.dx, J])
            X0 = MX.sym('X0', self.x.shape[0])
            U = MX.sym('U',
                       self.u.shape[0] + self.theta.shape[0] + self.dm.shape[0] + self.sp.shape[0] + self.uprev.shape[
                           0])
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT / 2 * k1, U)
                k3, k3_q = f(X + DT / 2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            self.F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

        # "Lift" initial conditions
        xk = MX.sym('x0', self.x.shape[0])  # first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        u0_sym = MX.sym('u0_par', self.u.shape[0])
        uk_prev = u0_sym

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # NLP
        self.w += [xk]
        self.w0 += list(xguess)
        self.lbw += list(lbx)
        self.ubw = list(ubx)
        self.g += [xk - x0_sym]
        self.lbg += list(np.zeros(self.dx.shape[0]))
        self.ubg += list(np.zeros(self.dx.shape[0]))

        # Check if the setpoints and targets are trajectories
        if not DRTO:
            spk = self.sp
            # targetk = self.target
        else:
            spk = MX.sym('SP_k', 2 * (N + 1))
            # targetk = MX.sym('Target_k', 2*N)

        # NLP build
        for k in range(0, self.N):

            # uk as decision variable
            uk = MX.sym('u_' + str(k), self.u.shape[0])
            self.w += [uk]
            self.lbw += list(lbu)
            self.ubw += list(ubu)
            self.w0 += list(uguess)
            self.g += [uk - uk_prev]  # delta_u

            # Control horizon
            if k >= self.M:
                self.lbg += list(np.zeros(self.u.shape[0]))
                self.ubg += list(np.zeros(self.u.shape[0]))
            else:
                self.lbg += list(lbdu)
                self.ubg += list(ubdu)

            # Loop over collocation points

            if not DRTO:  # check if the setpoints and targets are trajectories
                fi = self.F(x0=xk, p=vertcat(uk, self.theta, self.dm, spk, uk_prev))
            else:
                fi = self.F(x0=xk, p=vertcat(uk, self.theta, self.dm, vertcat(spk[k], spk[k + N + 1]), uk_prev))

            xk_end = fi['xf']

            self.J += fi['qf']  # add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            xk = MX.sym('x_' + str(k + 1), self.x.shape[0])
            self.w += [xk]
            self.lbw += list(lbx)
            self.ubw += list(ubx)
            self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk_end - xk]
            self.lbg += list(np.zeros(self.x.shape[0]))
            self.ubg += list(np.zeros(self.x.shape[0]))

            # u(k-1)
            uk_prev = copy.deepcopy(uk)

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym, u0_sym, self.theta, self.dm, spk)
        }  # nlp construction

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def calc_control_actions(self, x0, u0, sp, theta0=[], dm0=[], ksim=None):
        """
        Performs 1 optimization step for the NMPC
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(x0, u0, theta0, dm0, sp),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step ' + str(ksim) + ': NMPC solver did not converge.')
            else:
                print('Time step ' + str(ksim) + ': NMPC optimal solution found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step: NMPC solver did not converge.')
            else:
                print('Time step: NMPC optimal solution found.')

        # Solution
        wopt = sol['x'].full()
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x.shape[0]))
        uopt = np.zeros((self.N, self.u.shape[0]))
        for i in range(0, self.x.shape[0]):
            xopt[:, i] = wopt[i:: self.x.shape[0] + self.u.shape[0]].reshape(-1)  # optimal state
        for i in range(0, self.u.shape[0]):
            uopt[:, i] = wopt[self.x.shape[0] + i:: self.x.shape[0] + self.u.shape[0]].reshape(-1)  # optimal inputs

        # First control action
        uin = uopt[0, :]
        return {
            'x': xopt,
            'u': uopt,
            'U': uin
        }


class NMPC_SS:
    """
    This class creates an NMPC using casadi symbolic framework
    """

    def __init__(self, dt, N, M, Q, W, x, u, c, dx, theta, dm, xguess=None,
                 uguess=None, lbx=None, ubx=None, lbu=None, ubu=None, lbdu=None,
                 ubdu=None, DRTO=False, opts={}, casadi_integrator=False):
        self.dt = dt
        self.dx = dx
        self.x = x
        self.c = c
        self.u = u
        self.theta = theta
        self.N = N
        self.M = M
        self.dm = dm
        self.casadi_integrator = casadi_integrator

        '''
        if tgt: #evaluates tracking target inputs
             self.R = R
        else:
            self.R = np.zeros((self.u.shape[0], self.u.shape[0]))
        '''

        self.Q = Q
        self.W = W

        # Guesses
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        lbdu = -inf * np.ones(self.u.shape[0]) if lbdu is None else lbdu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu
        ubdu = -inf * np.ones(self.u.shape[0]) if ubdu is None else ubdu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in lbdu: lbdu = np.array([-inf if v is None else v for v in lbdu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])
        if None in ubdu: ubdu = np.array([-inf if v is None else v for v in ubdu])

        # Quadratic cost function
        self.sp = MX.sym('SP', self.c.shape[0])
        # self.target = MX.sym('Target', self.u.shape[0])
        self.uprev = MX.sym('u_prev', self.u.shape[0])
        J = (self.c - self.sp).T @ Q @ (self.c - self.sp) + (self.u - self.uprev).T @ W @ (self.u - self.uprev)

        if self.casadi_integrator:
            dae = {'x': self.x, 'p': vertcat(self.u, self.theta, self.dm, self.sp, self.uprev), 'ode': self.dx,
                   'quad': J}
            opts_dae = {'tf': self.dt}
            self.F = integrator('F', 'idas', dae, opts_dae)

        else:
            M = 4  # RK4 steps per interval
            DT = self.dt / M
            f = Function('f', [self.x, vertcat(self.u, self.theta, self.dm, self.sp, self.uprev)], [self.dx, J])
            X0 = MX.sym('X0', self.x.shape[0])
            U = MX.sym('U',
                       self.u.shape[0] + self.theta.shape[0] + self.dm.shape[0] + self.sp.shape[0] + self.uprev.shape[
                           0])
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT / 2 * k1, U)
                k3, k3_q = f(X + DT / 2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            self.F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

        # "Lift" initial conditions
        # xk = MX.sym('x0', self.x.shape[0]) #first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        u0_sym = MX.sym('u0_par', self.u.shape[0])
        uk_prev = u0_sym
        xk = x0_sym

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # NLP
        # self.w += [xk]
        # self.w0 += list(xguess)
        # self.lbw += list(lbx)
        # self.ubw = list(ubx)
        self.g += [xk]
        self.lbg += list(lbx)
        self.ubg += list(ubx)

        # Check if the setpoints and targets are trajectories
        if not DRTO:
            spk = self.sp
            # targetk = self.target
        else:
            spk = MX.sym('SP_k', 2 * (N + 1))
            # targetk = MX.sym('Target_k', 2*N)

        # NLP build
        for k in range(0, self.N):

            # uk as decision variable
            uk = MX.sym('u_' + str(k), self.u.shape[0])
            self.w += [uk]
            self.lbw += list(lbu)
            self.ubw += list(ubu)
            self.w0 += list(uguess)
            self.g += [uk - uk_prev]  # delta_u

            # Control horizon
            if k >= self.M:
                self.lbg += list(np.zeros(self.u.shape[0]))
                self.ubg += list(np.zeros(self.u.shape[0]))
            else:
                self.lbg += list(lbdu)
                self.ubg += list(ubdu)

            # Loop over collocation points

            if not DRTO:  # check if the setpoints and targets are trajectories
                fi = self.F(x0=xk, p=vertcat(uk, self.theta, self.dm, spk, uk_prev))
            else:
                fi = self.F(x0=xk, p=vertcat(uk, self.theta, self.dm, vertcat(spk[k], spk[k + N + 1]), uk_prev))

            xk = fi['xf']

            self.J += fi['qf']  # add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            # xk = MX.sym('x_' + str(k + 1), self.x.shape[0])
            # self.w += [xk]
            # self.lbw += list(lbx)
            # self.ubw += list(ubx)
            # self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk]
            self.lbg += list(lbx)
            self.ubg += list(ubx)

            # u(k-1)
            uk_prev = copy.deepcopy(uk)

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym, u0_sym, self.theta, self.dm, spk)
        }  # nlp construction

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def calc_control_actions(self, x0, u0, sp, theta0=[], dm0=[], ksim=None):
        """
        Performs 1 optimization step for the NMPC
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(x0, u0, theta0, dm0, sp),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step ' + str(ksim) + ': NMPC solver did not converge.')
            else:
                print('Time step ' + str(ksim) + ': NMPC optimal solution found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step: NMPC solver did not converge.')
            else:
                print('Time step: NMPC optimal solution found.')

        # Solution
        wopt = sol['x'].full()
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x.shape[0]))
        uopt = np.zeros((self.N, self.u.shape[0]))
        xopt[0, :] = np.array(x0).reshape(-1)

        for k in range(self.u.shape[0]):
            uopt[:, k] = wopt[k::self.u.shape[0]].reshape(-1)

        for i in range(len(uopt)):
            Fk = self.F(x0=xopt[i, :], p=vertcat(uopt[i, :], theta0, dm0, sp, u0))
            xopt[i + 1, :] = Fk['xf'].full().reshape(-1)
            u0 = uopt[i, :]

        # First control action
        uin = uopt[0, :]
        return {
            'x': xopt,
            'u': uopt,
            'U': uin
        }
