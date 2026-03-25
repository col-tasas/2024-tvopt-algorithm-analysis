"""State-space realizations for unconstrained optimization algorithms; each returns (G, p, q)."""

import numpy as np
import control as ctrl

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def gradient_descent(m, L, K=1):
    K     = K or 1
    alpha = 2.0 / (m + L)

    A, B, C, D = _lift(1, -alpha, 1, 0, K)
    return ctrl.ss(A, B, C, D, dt=1), K, 0


def nesterov(m, L, K=1):
    K     = K or 1
    alpha = 1.0 / L
    beta  = (np.sqrt(L / m) - 1.0) / (np.sqrt(L / m) + 1.0)

    A = np.array([[1 + beta, -beta], [1, 0]])
    B = np.array([[-alpha], [0]])
    C = np.array([[1 + beta, -beta]])

    A_K, B_K, C_K, D_K = _lift(A, B, C, 0, K)
    return ctrl.ss(A_K, B_K, C_K, D_K, dt=1), K, 0


def heavy_ball(m, L, K=1):
    K     = K or 1
    alpha = (2.0 / (np.sqrt(L) + np.sqrt(m))) ** 2
    beta  = ((np.sqrt(L / m) - 1.0) / (np.sqrt(L / m) + 1.0)) ** 2

    A = np.array([[1 + beta, -beta], [1, 0]])
    B = np.array([[-alpha], [0]])
    C = np.array([[1, 0]])

    A_K, B_K, C_K, D_K = _lift(A, B, C, 0, K)
    return ctrl.ss(A_K, B_K, C_K, D_K, dt=1), K, 0


def triple_momentum(m, L, K=1):
    K     = K or 1
    rho   = 1.0 - 1.0 / np.sqrt(L / m)
    alpha = (1.0 + rho) / L
    beta  = rho ** 2 / (2.0 - rho)
    gamma = rho ** 2 / ((1.0 + rho) * (2.0 - rho))

    A = np.array([[1 + beta, -beta], [1, 0]])
    B = np.array([[-alpha], [0]])
    C = np.array([[1 + gamma, -gamma]])

    A_K, B_K, C_K, D_K = _lift(A, B, C, 0, K)
    return ctrl.ss(A_K, B_K, C_K, D_K, dt=1), K, 0


def c2m(m, L, K=1):
    K     = K or 1
    kappa = L / m
    if kappa == 1.0:
        kappa = 1.0 + 1e-10
    rho = _c2m_rho(kappa)

    alpha = (1.0 - rho) ** 2 / m
    beta  = (rho / (kappa - 1.0)) * (1.0 - kappa * (1.0 - 3.0 * rho) / (1.0 + rho))
    eta   = (rho / (kappa - 1.0)) * ((1.0 + rho) / (1.0 - rho) ** 2 - kappa / (1.0 + rho))

    A = np.array([[1.0 + beta, -beta], [1.0, 0.0]])
    B = np.array([[-alpha], [0.0]])
    C = np.array([[1.0 + eta, -eta]])

    A_K, B_K, C_K, D_K = _lift(A, B, C, 0, K)
    return ctrl.ss(A_K, B_K, C_K, D_K, dt=1), K, 0


### parameters of c2m
def _c2m_rho(kappa):
    """Optimal C2M convergence rate for condition number kappa."""
    if kappa < 9.0 + 4.0 * np.sqrt(5.0):
        return (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    coeffs = [
        8 * kappa * (kappa + 1),
        -(23 * kappa ** 2 + 18 * kappa + 7),
        2 * (5 * kappa ** 2 - 14 * kappa - 7),
        31 * kappa ** 2 + 50 * kappa + 15,
        -4 * (11 * kappa ** 2 - 4 * kappa - 11),
        23 * kappa ** 2 - 30 * kappa + 23,
        -2 * (kappa - 1) * (3 * kappa + 1),
        (kappa - 1) ** 2,
    ]
    roots    = np.roots(coeffs)
    pos_real = [r.real for r in roots if r.real > 0 and abs(r.imag) < 1e-10]
    rho_c2m  = min(pos_real) if pos_real else 0.1
    return 0.5 * (rho_c2m + 1.0 - np.sqrt(2.0 / kappa))


### lifting
def _lift(A, B, C, D_scalar, K=1):
    """Lift a single-step realization (A, B, C, D) to its K-step multi-gradient variant."""
    if K is None:
        K = 1
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))

    if K == 1:
        D = np.atleast_2d(np.asarray(D_scalar, dtype=float))
        return A, B, C, D

    n = A.shape[0]
    A_pows = [np.eye(n)]
    for _ in range(K):
        A_pows.append(A_pows[-1] @ A)

    A_K = A_pows[K]
    B_K = np.hstack([A_pows[K - 1 - j] @ B for j in range(K)])
    C_K = np.vstack([C @ A_pows[i]         for i in range(K)])

    D_K = np.zeros((K, K))
    for i in range(K):
        for j in range(i):
            D_K[i, j] = float(C @ A_pows[i - 1 - j] @ B)

    return A_K, B_K, C_K, D_K