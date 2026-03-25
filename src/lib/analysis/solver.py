"""IQC LMI solvers for a single candidate rho; returns (feasible, result) pairs."""

import numpy as np
import cvxpy as cvx
import control as ctrl

from lib.analysis.lyapunov import PolynomialLyapunovMatrix
from lib.analysis.lure import build_lure_system, build_multiplier


def solve_static_iqc(algo, consistent_polytope, rho, optimize_bound=True, eps=1e-6, only_L=False):
    """Solve the Static IQC (Theorem 5.1) LMI at a given rho; returns (feasible, result)."""
    # Dimensions from a reference call
    _G0, _p0, _q0 = algo(1, 1)
    n_xi   = _G0.nstates
    I_n_xi = np.eye(n_xi)

    # --- SDP variables ---
    LMI_system = []
    lyap = PolynomialLyapunovMatrix(param_dim=1, poly_degree=2, n_eta=n_xi)
    Multiplier, _ = build_multiplier(_p0, _q0, vIQC=False)

    sigm   = cvx.Variable(1, nonneg=True)
    sigm_I = cvx.multiply(sigm, I_n_xi)

    # --- Grid over parameter space ---
    for p_k, delta_p in consistent_polytope:
        p_kp1 = p_k + delta_p

        P_k   = lyap.P(p_k)
        P_kp1 = lyap.P(p_kp1)

        if only_L:
            m, L = 1, p_k[0]
        else:
            m, L = p_k[0], p_k[1]

        _G, _p, _q = algo(m, L)

        G_hat, _ = build_lure_system(_G, m, L, _p, _q, vIQC=False)
        A_hat, B_hat, C_hat, D_hat = ctrl.ssdata(G_hat)

        CD = np.block([[C_hat, D_hat]])

        # Lyapunov decrease:  [A B]ᵀ P_{k+1} [A B]  −  diag(ρ²P_k, 0)  ≼ 0
        LMI_lyap = cvx.bmat([
            [A_hat.T @ P_kp1 @ A_hat - rho**2 * P_k, A_hat.T @ P_kp1 @ B_hat],
            [B_hat.T @ P_kp1 @ A_hat,                B_hat.T @ P_kp1 @ B_hat],
        ])
        # IQC multiplier term
        LMI_iqc = CD.T @ Multiplier @ CD

        LMI = LMI_lyap + LMI_iqc

        LMI_system.append(P_k   >> eps * I_n_xi)
        LMI_system.append(P_kp1 >> eps * I_n_xi)
        LMI_system.append(LMI   << 0)

        # Bound on condition number via Schur complement
        LMI_system.append(P_k   << sigm_I)
        LMI_system.append(P_kp1 << sigm_I)
        LMI_system.append(cvx.bmat([[P_k,   I_n_xi], 
                                    [I_n_xi, sigm_I]]) >> 0)
        LMI_system.append(cvx.bmat([[P_kp1, I_n_xi], 
                                    [I_n_xi, sigm_I]]) >> 0)

    cost    = cvx.Minimize(sigm) if optimize_bound else cvx.Minimize(0)
    problem = cvx.Problem(cost, LMI_system)

    try:
        problem.solve(solver=cvx.MOSEK)
    except cvx.SolverError:
        del lyap
        return False, None

    if problem.status == cvx.OPTIMAL:
        p_grid = list(zip(*consistent_polytope))[0]
        cond_P = lyap.condition_P(p_grid)
        del lyap
        return True, {'cond_P': cond_P}

    del lyap
    return False, None


def solve_var_iqc(algo, consistent_polytope, rho, *, optimize_bound=False, eps=1e-6, weights=None, only_L=False):
    """Solve the Varying IQC (Theorem 5.4) LMI at a given rho; returns (feasible, result)."""
    k1, k2, k3 = (1, 1, 1) if not weights else weights

    # Dimensions from a reference call
    _G0, _p0, _q0 = algo(1, 1)
    n_xi = _G0.nstates
    n_y  = _G0.noutputs
    n_u  = _G0.ninputs

    n_zeta  = 4 * _p0 + _q0
    n_eta0  = n_xi + n_zeta
    I_n_eta = np.eye(n_eta0)

    # --- SDP variables ---
    LMI_system = []
    lyap = PolynomialLyapunovMatrix(param_dim=1, poly_degree=2, n_eta=n_eta0)

    Multiplier, Variables = build_multiplier(_p0, _q0, vIQC=True)
    _, _, lambda_f, _ = Variables

    gamm_xi = cvx.Variable(1, nonneg=True)
    gamm_dd = cvx.Variable(1, nonneg=True)
    sigm       = cvx.Variable(1, nonneg=True)
    sigm_I     = cvx.multiply(sigm, I_n_eta)

    # --- Grid over parameter space ---
    for p_k, delta_p in consistent_polytope:
        p_kp1 = p_k + delta_p

        P_k   = lyap.P(p_k)
        P_kp1 = lyap.P(p_kp1)

        if only_L:
            m, L = 1, p_k[0]
        else:
            m, L = p_k[0], p_k[1]

        _G, _p, _q = algo(m, L)
        AG, BG, CG, DG = ctrl.ssdata(_G)

        # Augment plant with delta-model inputs: [g; Δξ*; Δδ]
        BG_aug = np.block([[BG, np.eye(n_xi), np.zeros((n_xi, _p))]])
        DG_aug = np.block([[DG, np.zeros((n_y, n_xi)), np.zeros((n_y, _p))]])
        G_aug  = ctrl.ss(AG, BG_aug, CG, DG_aug, dt=1)

        G_hat, _ = build_lure_system(G_aug, m, L, _p, _q, vIQC=True, rho=rho)
        A_hat, B_hat, C_hat, D_hat = ctrl.ssdata(G_hat)

        n_eta = A_hat.shape[0]   # = n_eta0
        n_psi = C_hat.shape[0]

        # --- LMI (5.15) ---
        LMI_inner = cvx.bmat([
            [-rho**2 * P_k,np.zeros((n_eta, n_eta)),np.zeros((n_eta, n_psi)),np.zeros((n_eta, n_xi)),np.zeros((n_eta, _p))],
            [np.zeros((n_eta, n_eta)),P_kp1,np.zeros((n_eta, n_psi)),np.zeros((n_eta, n_xi)),np.zeros((n_eta, _p))],
            [np.zeros((n_psi, n_eta)),np.zeros((n_psi, n_eta)),Multiplier,np.zeros((n_psi, n_xi)),np.zeros((n_psi, _p))],
            [np.zeros((n_xi, n_eta)),np.zeros((n_xi, n_eta)),np.zeros((n_xi, n_psi)),-cvx.multiply(gamm_xi, np.eye(n_xi)),np.zeros((n_xi, _p))],
            [np.zeros((_p, n_eta)),np.zeros((_p, n_eta)),np.zeros((_p, n_psi)),np.zeros((_p, n_xi)),-cvx.multiply(gamm_dd, np.eye(_p))],
        ])

        LMI_outer = cvx.bmat([
            [np.eye(n_eta), np.zeros((n_eta, n_u + n_xi + _p))],
            [np.block([[A_hat, B_hat]])],
            [np.block([[C_hat, D_hat]])],
            [np.zeros((n_xi, n_eta + n_u)),np.eye(n_xi),np.zeros((n_xi, _p))],
            [np.zeros((_p, n_eta + n_u + n_xi)),np.eye(_p)],
        ])

        LMI = LMI_outer.T @ LMI_inner @ LMI_outer

        LMI_system.append(P_k   >> eps * np.eye(n_eta))
        LMI_system.append(P_kp1 >> eps * np.eye(n_eta))
        LMI_system.append(LMI   << 0)

        LMI_system.append(P_k   << sigm_I)
        LMI_system.append(P_kp1 << sigm_I)
        LMI_system.append(cvx.bmat([[P_k,   I_n_eta], [I_n_eta, sigm_I]]) >> 0)
        LMI_system.append(cvx.bmat([[P_kp1, I_n_eta], [I_n_eta, sigm_I]]) >> 0)

    # --- Objective ---
    sensitivity_x = gamm_xi
    sensitivity_g = _p0 * gamm_dd

    if optimize_bound:
        cost = cvx.Minimize(
            sigm + k1 * cvx.sum(lambda_f) + k2 * sensitivity_x + k3 * sensitivity_g
        )
    else:
        cost = cvx.Minimize(0)

    problem = cvx.Problem(cost, LMI_system)

    try:
        problem.solve(solver=cvx.MOSEK)
    except cvx.SolverError:
        del lyap
        return False, None

    if problem.status == cvx.OPTIMAL:
        p_grid = list(zip(*consistent_polytope))[0]
        eig_min, eig_max = lyap.min_max_eigval(p_grid)
        c1 = eig_max / eig_min
        c2 = 1.0 / eig_min

        result = {
            'c1':            float(c1),
            'c2':            float(c2),
            'sensitivity_x': float(np.array(sensitivity_x.value).item()),
            'sensitivity_g': float(np.array(sensitivity_g.value).item()),
            'lambda_f':      np.asarray([l.value for l in lambda_f]),
            't':             float(sigm.value[0]),
        }
        del lyap
        return True, result

    del lyap
    return False, None