import numpy as np
import cvxpy as cvx
import control as ctrl
import scipy.linalg as linalg

from lib.lyapunov_matrix import PolynomialLyapunovMatrix
from lib.lure import build_lure_system, build_multiplier


def bisection_thm1(algo, consistent_polytope, optimize_bound=True, rho_max=1.5, eps=1e-6):
    sol = None

    # Get algorithm dimensions
    G, p, q = algo(1,1)
    n_xi = G.nstates

    I_n_xi = np.eye(n_xi)

    ### start bisection ###
    rho_min = 0
    rho_tol = 1e-3

    while (rho_max-rho_min > rho_tol):

        rho = (rho_min + rho_max)/2

        ### SDP variables ###
        LMI_system = list()
        
        lyap = PolynomialLyapunovMatrix(param_dim=1, poly_degree=2, n_eta=n_xi)

        Multiplier, Variables = build_multiplier(p, q, vIQC=False)
        
        t       = cvx.Variable(1, nonneg=True)
        t_I     = cvx.multiply(t, I_n_xi)

        ### grid over parameter space ###
        for p_k, delta_p in consistent_polytope:

            p_kp1 = p_k + delta_p

            P_k   = lyap.P(p_k)
            P_kp1 = lyap.P(p_kp1)

            ### algorithm ### TODO: update with lambda function m(p_k), L(p_k) instead of m=1, L=p_k
            m, L = 1, p_k[0]
            G, p, q = algo(m,L)

            G_hat, Psi = build_lure_system(G, m, L, p, q, vIQC=False)
            A_hat, B_hat, C_hat, D_hat = ctrl.ssdata(G_hat)
            
            ### LMI
            CD = np.block([[C_hat, D_hat]])

            LMI_lyap = cvx.bmat([[A_hat.T @ P_kp1 @ A_hat - rho**2 * P_k, A_hat.T @ P_kp1 @ B_hat],
                                 [B_hat.T @ P_kp1 @ A_hat,                B_hat.T @ P_kp1 @ B_hat]])
            LMI_iqc = CD.T @ Multiplier @ CD

            LMI = LMI_lyap + LMI_iqc

            LMI_system.append(P_k   >> eps*I_n_xi)
            LMI_system.append(P_kp1 >> eps*I_n_xi)
            LMI_system.append(LMI << 0)

            LMI_system.append(P_k   << t_I)
            LMI_system.append(P_kp1 << t_I)
            LMI_system.append(cvx.bmat([[P_k,   I_n_xi], [I_n_xi, t_I]]) >> 0)
            LMI_system.append(cvx.bmat([[P_kp1, I_n_xi], [I_n_xi, t_I]]) >> 0)

        # solve problem
        if optimize_bound:
            cost = cvx.Minimize(t)
        else:
            cost = cvx.Minimize(0)
            
        problem = cvx.Problem(cost, LMI_system)

        try:
            problem.solve(solver=cvx.MOSEK)
        except(cvx.SolverError):
            pass
    
        if problem.status == cvx.OPTIMAL:
            ### solution found, decrease rho, save solution
            rho_max = rho

            cond_P = lyap.condition_P(list(zip(*consistent_polytope))[0])

            sol = cond_P
        else:
            ### infeasible, increase rho
            rho_min = rho

        del lyap
        
    return rho_max, sol


def bisection_thm2(algo, consistent_polytope, optimize_bound=False, rho_max=1.5, eps=1e-6):

    sol = None

    # Get algorithm dimensions
    G, p, q = algo(1,1)
    n_xi = G.nstates
    n_y = G.noutputs
    n_u = G.ninputs

    ### get dimensions ###
    n_zeta = 4*p + q
    n_eta0 = n_xi + n_zeta

    I_n_eta = np.eye(n_eta0)

    ### start bisection ###
    rho_min = 0
    rho_tol = 1e-3

    while (rho_max-rho_min > rho_tol):

        rho = (rho_min + rho_max)/2

        ### SDP variables ###
        LMI_system = list()
        
        lyap = PolynomialLyapunovMatrix(param_dim=1, poly_degree=2, n_eta=n_eta0)

        Multiplier, Variables = build_multiplier(p, q, vIQC=True)
        lambd_ml_sector, lambd_zInf_sector, lambd_mL_offby1, lambd_zInf_offby1 = Variables
        
        gamm_xi = cvx.Variable(1, nonneg=True)
        gamm_dd = cvx.Variable(1, nonneg=True)
        t       = cvx.Variable(1, nonneg=True)
        t_I     = cvx.multiply(t, I_n_eta)

        ### grid over parameter space ###
        for p_k, delta_p in consistent_polytope:

            p_kp1 = p_k + delta_p

            P_k   = lyap.P(p_k)
            P_kp1 = lyap.P(p_kp1)

            ### algorithm ### TODO: update with lambda function m(p_k), L(p_k) instead of m=1, L=p_k
            m, L = 1, p_k[0]
            G, p, q = algo(m,L)
            AG, BG, CG, DG = ctrl.ssdata(G)

            ### augment plant with delta models ###
            BG_aug = np.block([[BG, np.eye(n_xi), np.zeros((n_xi,p))]])
            DG_aug = np.block([[DG, np.zeros((n_y, n_xi)), np.zeros((n_y,p))]])
            G_aug  = ctrl.ss(AG, BG_aug, CG, DG_aug, dt=1)

            G_hat, Psi = build_lure_system(G_aug, m, L, p, q, vIQC=True, rho=rho)
            A_hat, B_hat, C_hat, D_hat = ctrl.ssdata(G_hat)

            # get dimensions
            n_eta = A_hat.shape[0]
            n_psi = C_hat.shape[0]
            
            LMI_inner = cvx.bmat([
                [-rho**2 * P_k,              np.zeros((n_eta, n_eta)), np.zeros((n_eta, n_psi)), np.zeros((n_eta, n_xi)),             np.zeros((n_eta, p))],
                [np.zeros((n_eta, n_eta)),   P_kp1,                    np.zeros((n_eta, n_psi)), np.zeros((n_eta, n_xi)),             np.zeros((n_eta, p))],
                [np.zeros((n_psi, n_eta)),   np.zeros((n_psi, n_eta)), Multiplier,               np.zeros((n_psi, n_xi)),             np.zeros((n_psi, p))],
                [np.zeros((n_xi, n_eta)),    np.zeros((n_xi, n_eta)),  np.zeros((n_xi, n_psi)), -cvx.multiply(gamm_xi, np.eye(n_xi)), np.zeros((n_xi, p)) ],
                [np.zeros((p, n_eta)),       np.zeros((p, n_eta)),     np.zeros((p, n_psi)),     np.zeros((p, n_xi)),                -cvx.multiply(gamm_dd, np.eye(p))]
            ])

            LMI_outer = cvx.bmat([
                [np.eye(n_eta), np.zeros((n_eta, n_u + n_xi + p))],
                [np.block([[A_hat, B_hat]])],
                [np.block([[C_hat, D_hat]])],
                [np.zeros((n_xi, n_eta + n_u)), np.eye(n_xi), np.zeros((n_xi,p))],
                [np.zeros((p, n_eta + n_u + n_xi)), np.eye(p)]
            ])

            LMI = LMI_outer.T @ LMI_inner @ LMI_outer

            LMI_system.append(P_k   >> eps*np.eye(n_eta))
            LMI_system.append(P_kp1 >> eps*np.eye(n_eta))
            LMI_system.append(LMI << 0)

            LMI_system.append(P_k   << t_I)
            LMI_system.append(P_kp1 << t_I)
            LMI_system.append(cvx.bmat([[P_k,   I_n_eta], [I_n_eta, t_I]]) >> 0)
            LMI_system.append(cvx.bmat([[P_kp1, I_n_eta], [I_n_eta, t_I]]) >> 0)

        # solve problem
        sensitivity_x = gamm_xi
        sensitivity_f = cvx.sum(lambd_mL_offby1)
        sensitivity_g = p*gamm_dd
        if optimize_bound:
            cost = cvx.Minimize(t + sensitivity_f + sensitivity_x + sensitivity_g)
        else:
            cost = cvx.Minimize(0)
            
        problem = cvx.Problem(cost, LMI_system)

        try:
            problem.solve(solver=cvx.MOSEK)
        except(cvx.SolverError):
            pass
    
        if problem.status == cvx.OPTIMAL:
            ### solution found, decrease rho, save solution
            rho_max = rho

            eig_min, eig_max = lyap.min_max_eigval(list(zip(*consistent_polytope))[0])
            c1 = eig_max / eig_min
            c2 = 1 / eig_min

            sol = ((c1,c2), sensitivity_f.value, sensitivity_x.value, sensitivity_g.value)
        else:
            ### infeasible, increase rho
            rho_min = rho

        del lyap
        
    return rho_max, sol