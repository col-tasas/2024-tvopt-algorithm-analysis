"""State-space realizations for constrained optimization algorithms; each returns (G, p, q)."""

import numpy as np
import control as ctrl


# ---------------------------------------------------------------------------
# !!! Library not functional for constrained algorithms yet, stay tuned !!!
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def proximal_gradient(m, L):
    alpha = 2.0 / (m + L)

    A = 1
    B = np.asarray([[-alpha, -alpha]])
    C = np.asarray([[1], [1]])
    D = np.asarray([[0,      0     ],
                    [-alpha, -alpha]])

    return ctrl.ss(A, B, C, D, dt=1), 1, 1


def proximal_heavy_ball(m, L):
    alpha = (2.0 / (np.sqrt(L) + np.sqrt(m))) ** 2
    beta  = (np.sqrt(L / m) - 1.0) / (np.sqrt(L / m) + 1.0)

    A = np.asarray([[1 + beta, -beta], [1, 0]])
    B = np.asarray([[-alpha, -alpha], [0, 0]])
    C = np.asarray([[1, 0], [1 + beta, -beta]])
    D = np.asarray([[0, 0], [-alpha, -alpha]])

    return ctrl.ss(A, B, C, D, dt=1), 1, 1


def proximal_nesterov(m, L):
    alpha = 1.0 / L
    beta  = (np.sqrt(L / m) - 1.0) / (np.sqrt(L / m) + 1.0)

    A = np.asarray([[1 + beta, -beta], [1, 0]])
    B = np.asarray([[-alpha, -alpha], [0, 0]])
    C = np.asarray([[1 + beta, -beta], [1 + beta, -beta]])
    D = np.asarray([[0, 0], [-alpha, -alpha]])

    return ctrl.ss(A, B, C, D, dt=1), 1, 1


def proximal_triple_momentum(m, L):
    rho   = 1.0 - 1.0 / np.sqrt(L / m)
    alpha = (1.0 + rho) / L
    beta  = rho ** 2 / (2.0 - rho)
    gamma = rho ** 2 / ((1.0 + rho) * (2.0 - rho))

    A = np.asarray([[1 + beta, -beta], [1, 0]])
    B = np.asarray([[-alpha, -alpha], [0, 0]])
    C = np.asarray([[1 + gamma, -gamma], [1 + gamma, -gamma]])
    D = np.asarray([[0, 0], [-alpha, -alpha]])

    return ctrl.ss(A, B, C, D, dt=1), 1, 1


def accelerated_ogd(m, L):
    alpha = 1.0 / L
    gamma = 1.0 / L
    tau   = gamma * alpha

    A = np.asarray([[tau, 1 - tau], [0, 1]])
    B = np.asarray([[-gamma, -gamma, 0], [-alpha, 0, -alpha]])
    C = np.asarray([[tau, 1 - tau], [tau, 1 - tau], [0, 1]])
    D = np.asarray([[0, 0, 0], [-gamma, -gamma, 0], [-alpha, 0, -alpha]])

    return ctrl.ss(A, B, C, D, dt=1), 1, 2


def projected_ogd(m, L, K=1):
    gamma = 1.0 / L
    K = K or 1

    if K == 1:
        A = 1
        B = np.asarray([[-gamma, -gamma]])
        C = np.asarray([[1], [1]])
        D = np.asarray([[0, 0], [-gamma, -gamma]])
    else:
        A, B, C, D = _lift_ogd(gamma, K)

    return ctrl.ss(A, B, C, D, dt=1), K, K
