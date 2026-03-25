"""Tracking error simulation and IQC-based certified upper bound computation."""

import numpy as np
from lib.simulation.algorithm import Algorithm

# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def simulate(algo_def, objective, x0, T, certificate):
    """Run one simulation and return (tracking_error, error_bound) over T steps using SDP certificates."""
    algo = Algorithm(algo_def, nx=objective.nx)
    xi_0 = x0.copy()[:algo.internal_state_dim]
    algo.initialize(xi_0)
    xi_0 = algo.internal_state

    K_steps = algo.p   # number of gradient evaluation points per macro-step
    nx      = objective.nx

    objective.update(0)
    x_star_0, _, _ = objective.get_objective_info()
    xi_star_0  = np.tile(x_star_0, (algo.internal_state_dim // nx, 1))
    xi_delta_0 = xi_0 - xi_star_0

    tracking_error = []
    error_bound    = []
    Delta_xi_star  = []   # list of (n_xi, 1) vectors
    Delta_delta    = []   # list of K_steps-length lists of (nx, 1) vectors
    Delta_f_hat    = []   # list of K_steps-length lists of scalars

    grad_km1_s_km1 = [np.zeros((nx, 1)) for _ in range(K_steps)]
    y_km1          = np.tile(x_star_0, (K_steps, 1))  # (K_steps*nx, 1)
    x_star_km1     = x_star_0.copy()
    xi_star_km1    = xi_star_0.copy()
    m_km1 = L_km1  = None

    for k in range(T):
        objective.update(k)
        x_star_k, m_k, L_k = objective.get_objective_info()

        algo.update_sectors(m_k, L_k)
        algo.update_gradient(lambda x: objective.gradient(x))

        xi_k, x_k, y_k = algo.step()  # y_k: (K_steps*nx, 1)

        if hasattr(objective, 'name') and objective.name == 'robotic_ellipse_tracking':
            objective.update_joint_angles(x_k)

        # Gradient at each intermediate point y_k_i under time k
        grad_k_s_k = [objective.gradient(y_k[i*nx:(i+1)*nx]) for i in range(K_steps)]

        if k >= 1:
            # Gradient at each y_km1_i evaluated under current time k
            grad_k_s_km1 = [objective.gradient(y_km1[i*nx:(i+1)*nx]) for i in range(K_steps)]
            # Gradient variation at each intermediate point
            Delta_delta.append([
                grad_km1_s_km1[i] - grad_k_s_km1[i]
                for i in range(K_steps)
            ])

        # xi_star_k must be computed here — after the first if k>=1 block
        # and before the second one, so dxi = xi_star_km1 - xi_star_k is correct
        xi_star_k = np.tile(x_star_k, (algo.internal_state_dim // nx, 1))
        tracking_error.append(float(np.linalg.norm(xi_k - xi_star_k)))

        if k >= 1:
            dxi = xi_star_km1 - xi_star_k
            Delta_xi_star.append(dxi)

            # Function variation at each intermediate point y_km1_i
            delta_f_list = []
            for i in range(K_steps):
                yi = y_km1[i*nx:(i+1)*nx]
                f_hat_km1_i = (objective.eval(yi, prev_t=True)
                               - objective.eval(x_star_km1, prev_t=True))
                f_hat_k_i   = (objective.eval(yi)
                               - objective.eval(x_star_k))
                delta_f_list.append(
                    (L_k - m_k) * f_hat_k_i - (L_km1 - m_km1) * f_hat_km1_i
                )
            Delta_f_hat.append(delta_f_list)

        # --- Compute bound ---
        if certificate["iqcType"] == "static":
            error_bound_k = certificate["c"] * certificate["rho"]**k * np.linalg.norm(xi_delta_0)
            for t_idx in range(1, k + 1):
                error_bound_k += (certificate["c"]
                                  * certificate["rho"]**(k - t_idx)
                                  * np.linalg.norm(Delta_xi_star[t_idx - 1]))
            error_bound.append(float(error_bound_k))

        elif certificate["iqcType"] == "variational":
            term1 = (certificate["c1"]
                     * certificate["rho"]**(2*k)
                     * float(np.linalg.norm(xi_delta_0)**2))

            term2 = 0.0
            for t_idx in range(1, k + 1):
                delta_xi_star = Delta_xi_star[t_idx - 1]
                # Sum squared norms over all K intermediate gradient-variation vectors
                delta_gradient_norm_sq = sum(
                    float(np.linalg.norm(Delta_delta[t_idx - 1][i])**2)
                    for i in range(K_steps)
                )
                term2 += (certificate["c2"]
                          * certificate["rho"]**(2*(k - t_idx))
                          * (certificate["sensitivity_x"] * float(np.linalg.norm(delta_xi_star)**2)
                             + certificate["sensitivity_g"] * delta_gradient_norm_sq))

            term3 = 0.0
            for idx_p in range(K_steps):
                for t_idx in range(k - 1):
                    term3 += (certificate["c2"]
                              * certificate["rho"]**(2*(k - t_idx))
                              * certificate["lambda_f"][idx_p]
                              * float(Delta_f_hat[t_idx][idx_p]))

            error_bound.append(float(np.sqrt(term1 + term2 + term3)))

        # Advance previous-step quantities
        xi_star_km1    = xi_star_k
        y_km1          = y_k
        grad_km1_s_km1 = grad_k_s_k
        x_star_km1     = x_star_k
        m_km1, L_km1   = m_k, L_k

    return tracking_error, error_bound
