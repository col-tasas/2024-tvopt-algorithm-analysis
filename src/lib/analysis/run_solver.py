"""Bisection and grid-sweep wrappers around the IQC LMI solvers."""

import numpy as np
from tqdm import tqdm

from lib.analysis.solver import solve_static_iqc, solve_var_iqc


def static_IQC_rho_bisection(algo, polytope, optimize_bound=False, rho_max=1.5, eps=1e-6, only_L=False):
    """Binary search for the minimum feasible rho under the Static IQC LMI (Theorem 5.2)."""
    best_certificate = None
    rho_min          = 0.0
    rho_tol          = 1e-3

    while rho_max - rho_min > rho_tol:
        rho = (rho_min + rho_max) / 2
        feasible, solver_output = solve_static_iqc(algo, polytope, rho, optimize_bound=optimize_bound, eps=eps, only_L=only_L)
        if feasible:
            rho_max = rho
            best_certificate = {
                'rho':     np.round(rho, 4),
                'c':       solver_output['cond_P'],
                'iqcType': 'static'
            }
        else:
            rho_min = rho

    return best_certificate


def var_IQC_rho_bisection(algo, polytope, optimize_bound=False, rho_max=1.5, eps=1e-6, only_L=False):
    """Binary search for the minimum feasible rho under the Varying IQC LMI (Theorem 5.3)."""

    best_certificate = None
    rho_min          = 0.0
    rho_tol          = 1e-3

    while rho_max - rho_min > rho_tol:
        rho = (rho_min + rho_max) / 2
        feasible, solver_output = solve_var_iqc(algo, polytope, rho, optimize_bound=optimize_bound, eps=eps, only_L=only_L)
        if feasible:
            rho_max          = rho
            best_certificate = {      
                'rho':           np.round(rho, 4),                                       
                'c1':            solver_output['c1'],                             
                'c2':            solver_output['c2'],                             
                'sensitivity_x': solver_output['sensitivity_x'],                  
                'sensitivity_g': solver_output['sensitivity_g'],                  
                'lambda_f':      solver_output['lambda_f'],      
                'iqcType':       'variational'
            }
        else:
            rho_min = rho

    return best_certificate


def static_IQC_rho_sweep(algo, polytope, optimize_bound=True, rho_lo=0.0, rho_hi=1.0, rho_step=0.01, eps=1e-6, only_L=False):
    """Solve the Static IQC LMI at every rho on a uniform grid; returns one dict per feasible rho."""
    certificate_list = []
    rho_grid         = np.arange(rho_lo, rho_hi + 1e-12, rho_step)

    for rho in tqdm(rho_grid, desc='Static IQC: compute certificates over rho grid'):
        feasible, solver_output = solve_static_iqc(algo, polytope, rho, optimize_bound=optimize_bound, eps=eps, only_L=only_L)
        if feasible:
            certificate_list.append({
                'rho':     np.round(rho, 4),
                'c':       solver_output['cond_P'],
                'iqcType': 'static'
            })

    return certificate_list


def var_IQC_rho_sweep(algo, polytope, optimize_bound=True, rho_lo=0.0, rho_hi=1.0, rho_step=0.01, eps=1e-6, only_L=False):
    """Solve the Varying IQC LMI at every rho on a uniform grid; returns one dict per feasible rho."""
    certificate_list = []
    rho_grid         = np.arange(rho_lo, rho_hi + 1e-12, rho_step)

    for rho in tqdm(rho_grid, desc='Variational IQC: compute certificates over rho grid'):
        feasible, solver_output = solve_var_iqc(algo, polytope, rho, optimize_bound=optimize_bound, eps=eps, only_L=only_L)
        if feasible:
            certificate_list.append({
                'rho':           np.round(rho, 4),                                       
                'c1':            solver_output['c1'],                             
                'c2':            solver_output['c2'],                             
                'sensitivity_x': solver_output['sensitivity_x'],                  
                'sensitivity_g': solver_output['sensitivity_g'],                  
                'lambda_f':      solver_output['lambda_f'],      
                'iqcType':       'variational'
            })

    return certificate_list
