import numpy as np
import control as ctrl
import cvxpy as cvx
from scipy import linalg


def lti_stack(sys1, sys2):
    """
    Stacks two Linear Time-Invariant (LTI) systems into a single system.

    Returns:
    - A new LTI system that represents the stacked combination of sys1 and sys2.

    Raises:
    - ValueError: If the two systems do not have the same number of inputs.
    """
    
    A1, B1, C1, D1 = ctrl.ssdata(sys1)
    A2, B2, C2, D2 = ctrl.ssdata(sys2)

    if B1.shape[1] != B2.shape[1] or D1.shape[1] != D2.shape[1]:
        raise ValueError('Error in system stacking: number of inputs must be the same for both subsystems!')

    A = linalg.block_diag(A1, A2)
    B = np.vstack((B1, B2))
    C = linalg.block_diag(C1, C2)
    D = np.vstack((D1, D2))

    return ctrl.ss(A, B, C, D, dt=1)


def build_input_mapping(p, q, *, vIQC=False, n_xi=None):
    """
    Builds a matrix T such that:
    u_stacked = T @ u_true

    Inputs:
    - p: number of Ψ_i blocks
    - q: number of Ψ_j blocks
    - vIQC: 
        True  → Ψ_i: [s_i, δ_i, Δξ*, Δδ_i], Ψ_j: [z_j, g_j, Δξ*]
        False → Ψ_i: [s_i, δ_i],            Ψ_j: [z_j, g_j]
    - n_xi: dimension of Δξ*

    True input ordering:
      if vIQC=True:
        [s1..sp, z1..zq, δ1..δp, g1..gq, Δξ*, Δδ1..Δδp]
      if vIQC=False:
        [s1..sp, z1..zq, δ1..δp, g1..gq]
    """
    if vIQC and n_xi is None:
        raise ValueError("n_xi must be provided when vIQC=True")

    if vIQC:
        n_inputs_true = p + q + p + q + n_xi + p
        n_inputs_stacked = p * (3 + n_xi) + q * (2 + n_xi)
    else:
        n_inputs_true = p + q + p + q
        n_inputs_stacked = 2 * (p + q)

    T = np.zeros((n_inputs_stacked, n_inputs_true))
    offset = 0

    # Psi_i blocks
    for i in range(p):
        s_i, d_i = i, p + q + i
        if vIQC:
            xi_start, dd_i = 2 * p + 2 * q, 2 * p + 2 * q + n_xi + i
            T[offset:offset+2, [s_i, d_i]] = np.eye(2)
            T[offset+2:offset+2+n_xi, xi_start:xi_start+n_xi] = np.eye(n_xi)
            T[offset+2+n_xi, dd_i] = 1
            offset += 3 + n_xi
        else:
            T[offset:offset+2, [s_i, d_i]] = np.eye(2)
            offset += 2

    # Psi_j blocks
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
    """
    Builds a matrix S such that:
    y_reordered = S @ y_stacked

    Assumes that ordering is
    [Ψ_sector_mL_1, Ψ_offby1_mL_1, ... Ψ_sector_mL_p, Ψ_offby1_mL_p, ...
        Ψ_sector_0∞_1, Ψ_offby1_0∞_1, ... Ψ_sector_0∞_q, Ψ_offby1_0∞_q]

    Transforms ordering to
    [Ψ_sector_mL_1:p, Ψ_sector_0∞_1:q, Ψ_offby1_mL_1:p, Ψ_offby1_0∞_1:q]
    """
    total_outputs = p * (n_sector_i + n_offby1_i) + q * (n_sector_j + n_offby1_j)
    total_sector  = p * n_sector_i + q * n_sector_j

    S = np.zeros((total_outputs, total_outputs))

    current_input_index = 0
    sector_out = 0
    offby1_out = total_sector  # start writing offby1 outputs here

    for i in range(p):
        # Sector outputs
        for k in range(n_sector_i):
            S[sector_out, current_input_index + k] = 1
            sector_out += 1
        # Off-by-1 outputs
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
    """
    Builds an Integral Quadratic Constraint (IQC) for a given system.

    Parameters:
    - m: Lower bound for the sector condition.
    - L: Upper bound for the sector condition.
    - p: Number of inputs/outputs for the [m, L]-sector nonlinearity.
    - q: Number of inputs/outputs for the [0, ∞]-sector nonlinearity.
    - vIQC: determines whether an off-by-one vIQC is incorporated.
    - C: needed if vIQC=True to map Δx* to Δξ*

    Returns:
    - An LTI system representing the IQC.
    """
    if vIQC and (C is None or rho is None):
        raise ValueError("C and rho must be provided when vIQC=True")

    n_xi = C.shape[1] if vIQC else None

    ### IQC for i=1,...,p (gradient)
    Psi_i_list = list()
    for i in range(p):

        if vIQC:
            # dynamic IQC
            a = np.sqrt(m*(L-m)/2)
            Ci = C[i,:]
            Z_1xi = np.zeros((1,n_xi))

            # filter matrices
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
                              [np.zeros((4,3+n_xi))]])

            Psi_sector_mL_i = ctrl.ss([], [], [], D_psi[:2,:], dt=1)
            Psi_offby1_mL_i = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1)

            Psi_i = lti_stack(Psi_sector_mL_i, Psi_offby1_mL_i)
        else:
            # static IQC
            Psi_i = ctrl.ss([], [], [], np.asarray([[L, -1,],[-m, 1]]), dt=1)
        
        Psi_i_list.append(Psi_i)


    ### IQC for j=1,...,q (normal cone)
    Psi_j_list = list()
    for j in range(q):

        if vIQC:
            # dynamic IQC
            Ci = C[p+j,:]
            A_psi = 0
            B_psi = np.block([[1, 0, Ci]])
            C_psi = np.asarray([[-1], 
                                [0]])
            D_psi = np.block([[1, 0, Z_1xi], 
                              [0, 1, Z_1xi]])
            Psi_sector_zInf_j = ctrl.ss([], [], [], D_psi, dt=1)
            Psi_offby1_zInf_j = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1)

            Psi_j = lti_stack(Psi_sector_zInf_j, Psi_offby1_zInf_j)
        else:
            # static IQC
            Psi_j = ctrl.ss([], [], [], np.asarray([[0, 1], [1, 0]]), dt=1)

        Psi_j_list.append(Psi_j)    

    ### stack all IQCs
    Psi_all = ctrl.append(*Psi_i_list, *Psi_j_list)

    ### map to inputs:  [s1..sp, z1..zq, δ1..δp, g1..gq, xi*, dd1..ddp]
    ### map to outputs: (p + q) * psi_sector + (p + q) * psi_vIQC
    T = build_input_mapping(p, q, vIQC=vIQC, n_xi=n_xi)
    S = build_output_mapping(p, q)

    # Apply permutation to inputs
    A_permuted = Psi_all.A
    B_permuted = Psi_all.B @ T
    C_permuted = S @ Psi_all.C     if vIQC else Psi_all.C
    D_permuted = S @ Psi_all.D @ T if vIQC else Psi_all.D @ T

    Psi = ctrl.ss(A_permuted, B_permuted, C_permuted, D_permuted, dt=1)

    return Psi


def build_multiplier(p, q, vIQC):
    """
    Builds a block diagonal cvxpy expression with scalar lambda multipliers.

    Assumes that ordering is
    [Ψ_sector_mL_1:p, Ψ_sector_0∞_1:q, Ψ_offby1_mL_1:p, Ψ_offby1_0∞_1:q]
    
    Returns:
      Multiplier_cvx: cvxpy expression
      Variables: 
        - list of cvxpy variables 
        - [λ_sector_mL_1:p, λ_sector_0∞_1:q, λ_offby1_mL_1:p, λ_offby1_0∞_1:q]
    """
    M_mix = np.asarray([[0,1],
                        [1,0]])
    M_diff = np.asarray([[1,0],
                         [0,-1]])
    M2 = M_mix
    M6 = linalg.block_diag(1/2*M_mix, 
                           M_diff, 
                           1/2*M_diff) 

    total_blocks = p + q + p + q if vIQC else p + q

    lambdas_p_sec = [cvx.Variable(nonneg=True, name=f'lambda_p_sec_{i}') for i in range(p)]
    lambdas_q_sec = [cvx.Variable(nonneg=True, name=f'lambda_q_sec_{i}') for i in range(q)]

    Variables = [lambdas_p_sec, lambdas_q_sec]
    
    blocks = []
    
    # Append p sector blocks
    for lam in lambdas_p_sec:
        blocks.append(lam * M2)
    
    # Append q sector blocks
    for lam in lambdas_q_sec:
        blocks.append(lam * M2)
    
    if vIQC:
        lambdas_p_offby1 = [cvx.Variable(nonneg=True, name=f'lambda_p_offby1_{i}') for i in range(p)]
        lambdas_q_offby1 = [cvx.Variable(nonneg=True, name=f'lambda_q_offby1_{i}') for i in range(q)]

        Variables += [lambdas_p_offby1, lambdas_q_offby1]

        # Append p offby1 blocks
        for lam in lambdas_p_offby1:
            blocks.append(lam * M6)
        
        # Append q offby1 blocks
        for lam in lambdas_q_offby1:
            blocks.append(lam * M2)
    
    total_blocks = len(blocks)
    
    # Build big block diagonal cvxpy matrix
    Multiplier = cvx.bmat(
        [[blocks[i] if i == j else
          np.zeros((blocks[i].shape[0], blocks[j].shape[1]))
          for j in range(total_blocks)] for i in range(total_blocks)]
    )

    return Multiplier, Variables


def build_lure_system(G, m, L, p, q, vIQC=True, rho=None):
    """
    Builds a Lur'e system representing the interconnection of algorithm and monotone operator.

    Parameters:
    - G: The plant model in state-space form.
    - m: Lower bound for the sector condition.
    - L: Upper bound for the sector condition.
    - p: Number of inputs/outputs for the [m, L] operator.
    - q: Number of inputs/outputs for the [0, ∞] operator.
    - vIQC: determines whether a variational vIQC is incorporated.

    Returns:
    - A tuple (G_hat, Psi, Multiplier, Variables) representing 
        - G_hat: augmented system ready to be edployed in LMI
        - Psi: IQC that is used to build G_hat
        - Multiplier: Multiplier for outputs of G_hat
        - Variables: cvxpy multiplier used to build Multiplier
    """

    ### build IQC filter
    if vIQC:
        if rho is None:
            raise ValueError("rho must be provided when vIQC=True")

        Psi = build_iqc(m, L, p, q, vIQC=True, C=G.C, rho=rho)
    else:
        Psi = build_iqc(m, L, p, q, vIQC=False)
    n_psi = Psi.nstates
    
    ### build augmented plant
    # 1) iqc_input = [y; u] = [G*u; u] = [G; I]*u => build [G; I]
    n_in = G.ninputs 
    G_I = lti_stack(G, np.eye(n_in))
    
    # 2) psi = Psi * iqc_input => build Psi*[G; I]
    G_hat = ctrl.series(G_I, Psi)

    return G_hat, Psi 