"""LTI building blocks for the IQC/Lure framework: filters, multipliers, and augmented plants."""

import numpy as np
import control as ctrl
import cvxpy as cvx
from scipy import linalg


def lti_stack(sys1, sys2):
    """Stack two LTI systems with shared inputs; outputs are [y1; y2]."""
    A1, B1, C1, D1 = ctrl.ssdata(sys1)
    A2, B2, C2, D2 = ctrl.ssdata(sys2)

    if B1.shape[1] != B2.shape[1] or D1.shape[1] != D2.shape[1]:
        raise ValueError(
            'Error in system stacking: number of inputs must be the same for both subsystems!'
        )

    A = linalg.block_diag(A1, A2)
    B = np.vstack((B1, B2))
    C = linalg.block_diag(C1, C2)
    D = np.vstack((D1, D2))

    return ctrl.ss(A, B, C, D, dt=1)


def build_input_mapping(p, q, *, vIQC=False, n_xi=None):
    """Build the permutation matrix T satisfying u_stacked = T @ u_true for the IQC filter inputs."""
    if vIQC and n_xi is None:
        raise ValueError("n_xi must be provided when vIQC=True")

    if vIQC:
        n_inputs_true    = p + q + p + q + n_xi + p
        n_inputs_stacked = p * (3 + n_xi) + q * (2 + n_xi)
    else:
        n_inputs_true    = p + q + p + q
        n_inputs_stacked = 2 * (p + q)

    T = np.zeros((n_inputs_stacked, n_inputs_true))
    offset = 0

    for i in range(p):
        s_i, d_i = i, p + q + i
        if vIQC:
            xi_start = 2 * p + 2 * q
            dd_i     = 2 * p + 2 * q + n_xi + i
            T[offset:offset+2, [s_i, d_i]] = np.eye(2)
            T[offset+2:offset+2+n_xi, xi_start:xi_start+n_xi] = np.eye(n_xi)
            T[offset+2+n_xi, dd_i] = 1
            offset += 3 + n_xi
        else:
            T[offset:offset+2, [s_i, d_i]] = np.eye(2)
            offset += 2

    for j in range(q):
        z_j, g_j = p + j, p + q + p + j
        if vIQC:
            xi_start = 2 * p + 2 * q
            T[offset:offset+2, [z_j, g_j]] = np.eye(2)
            T[offset+2:offset+2+n_xi, xi_start:xi_start+n_xi] = np.eye(n_xi)
            offset += 2 + n_xi
        else:
            T[offset:offset+2, [z_j, g_j]] = np.eye(2)
            offset += 2

    return T


def build_output_mapping(p, q, n_sector_i=2, n_offby1_i=6, n_sector_j=2, n_offby1_j=2):
    """Build the permutation matrix S satisfying y_reordered = S @ y_stacked for IQC filter outputs."""
    total_outputs = p * (n_sector_i + n_offby1_i) + q * (n_sector_j + n_offby1_j)
    total_sector  = p * n_sector_i + q * n_sector_j

    S = np.zeros((total_outputs, total_outputs))

    current_input_index = 0
    sector_out  = 0
    offby1_out  = total_sector

    for i in range(p):
        for k in range(n_sector_i):
            S[sector_out, current_input_index + k] = 1
            sector_out += 1
        for k in range(n_offby1_i):
            S[offby1_out, current_input_index + n_sector_i + k] = 1
            offby1_out += 1
        current_input_index += n_sector_i + n_offby1_i

    for j in range(q):
        for k in range(n_sector_j):
            S[sector_out, current_input_index + k] = 1
            sector_out += 1
        for k in range(n_offby1_j):
            S[offby1_out, current_input_index + n_sector_j + k] = 1
            offby1_out += 1
        current_input_index += n_sector_j + n_offby1_j

    return S


def build_iqc(m, L, p, q, *, vIQC=False, C=None, rho=None):
    """Construct the IQC filter Psi for sector bounds (m, L) with p gradient and q normal-cone channels."""
    if vIQC and (C is None or rho is None):
        raise ValueError("C and rho must be provided when vIQC=True")

    n_xi = C.shape[1] if vIQC else None

    # --- IQC for i = 1, …, p  (gradient / [m,L]-sector nonlinearity) ---
    Psi_i_list = []
    for i in range(p):
        if vIQC:
            a    = np.sqrt(m * (L - m) / 2)
            Ci   = C[i, :]
            Z_1xi = np.zeros((1, n_xi))

            A_psi = np.zeros((4, 4))
            B_psi = np.block([[1,  0,    Ci,  0],
                               [0,  1, Z_1xi, -1],
                               [a,  0, Z_1xi,  0],
                               [-m, 1, Z_1xi,  0]])
            C_psi = np.asarray([[-L*rho**2, rho**2, 0,    0],
                                 [0,         0,      0,    0],
                                 [0,         0,      rho,  0],
                                 [rho*a,     0,      0,    0],
                                 [0,         0,      0,  rho],
                                 [-m*rho,  rho,      0,    0]])
            D_psi = np.block([[L, -1, Z_1xi, 0],
                               [-m, 1, Z_1xi, 0],
                               [np.zeros((4, 3 + n_xi))]])

            Psi_sector_mL_i = ctrl.ss([], [], [], D_psi[:2, :], dt=1)
            Psi_offby1_mL_i = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1)
            Psi_i = lti_stack(Psi_sector_mL_i, Psi_offby1_mL_i)
        else:
            Psi_i = ctrl.ss([], [], [], np.asarray([[L, -1], [-m, 1]]), dt=1)

        Psi_i_list.append(Psi_i)

    # --- IQC for j = 1, …, q  (normal cone / [0,∞]-sector nonlinearity) ---
    Psi_j_list = []
    for j in range(q):
        if vIQC:
            Ci   = C[p + j, :]
            Z_1xi = np.zeros((1, n_xi))

            A_psi = 0
            B_psi = np.block([[1, 0, Ci]])
            C_psi = np.asarray([[-1], [0]])
            D_psi = np.block([[1, 0, Z_1xi],
                               [0, 1, Z_1xi]])

            Psi_sector_zInf_j = ctrl.ss([], [], [], D_psi, dt=1)
            Psi_offby1_zInf_j = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1)
            Psi_j = lti_stack(Psi_sector_zInf_j, Psi_offby1_zInf_j)
        else:
            Psi_j = ctrl.ss([], [], [], np.asarray([[0, 1], [1, 0]]), dt=1)

        Psi_j_list.append(Psi_j)

    # --- Stack all IQCs and apply permutations ---
    Psi_all = ctrl.append(*Psi_i_list, *Psi_j_list)

    T = build_input_mapping(p, q, vIQC=vIQC, n_xi=n_xi)
    S = build_output_mapping(p, q)

    A_perm = Psi_all.A
    B_perm = Psi_all.B @ T
    C_perm = S @ Psi_all.C     if vIQC else Psi_all.C
    D_perm = S @ Psi_all.D @ T if vIQC else Psi_all.D @ T

    return ctrl.ss(A_perm, B_perm, C_perm, D_perm, dt=1)


def build_multiplier(p, q, vIQC):
    """Build the block-diagonal CVXPY multiplier and return (Multiplier, Variables)."""
    M_mix  = np.asarray([[0, 1], [1, 0]])
    M_diff = np.asarray([[1, 0], [0, -1]])
    M2 = M_mix
    M6 = linalg.block_diag(0.5 * M_mix, M_diff, 0.5 * M_diff)

    lambdas_p_sec = [cvx.Variable(nonneg=True, name=f'lambda_p_sec_{i}') for i in range(p)]
    lambdas_q_sec = [cvx.Variable(nonneg=True, name=f'lambda_q_sec_{i}') for i in range(q)]

    Variables = [lambdas_p_sec, lambdas_q_sec]
    blocks    = []

    for lam in lambdas_p_sec:
        blocks.append(lam * M2)
    for lam in lambdas_q_sec:
        blocks.append(lam * M2)

    if vIQC:
        lambdas_p_offby1 = [
            cvx.Variable(nonneg=True, name=f'lambda_p_offby1_{i}') for i in range(p)
        ]
        lambdas_q_offby1 = [
            cvx.Variable(nonneg=True, name=f'lambda_q_offby1_{i}') for i in range(q)
        ]
        Variables += [lambdas_p_offby1, lambdas_q_offby1]
        for lam in lambdas_p_offby1:
            blocks.append(lam * M6)
        for lam in lambdas_q_offby1:
            blocks.append(lam * M2)

    n_blocks = len(blocks)
    Multiplier = cvx.bmat(
        [[blocks[i] if i == j
          else np.zeros((blocks[i].shape[0], blocks[j].shape[1]))
          for j in range(n_blocks)]
         for i in range(n_blocks)]
    )

    return Multiplier, Variables


def build_lure_system(G, m, L, p, q, vIQC=True, rho=None):
    """Build the augmented Lure plant G_hat = Psi(G; I) and return (G_hat, Psi)."""
    if vIQC:
        if rho is None:
            raise ValueError("rho must be provided when vIQC=True")
        Psi = build_iqc(m, L, p, q, vIQC=True, C=G.C, rho=rho)
    else:
        Psi = build_iqc(m, L, p, q, vIQC=False)

    n_in = G.ninputs
    G_I  = lti_stack(G, np.eye(n_in))
    G_hat = ctrl.series(G_I, Psi)

    return G_hat, Psi
