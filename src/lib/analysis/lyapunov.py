"""Parameter-dependent polynomial Lyapunov matrix P(p) with CVXPY SDP variables."""

import numpy as np
import cvxpy as cvx
from itertools import combinations_with_replacement


class PolynomialLyapunovMatrix:
    """Polynomial Lyapunov matrix P(p) = sum P_alpha * p^alpha with symmetric CVXPY variables."""

    def __init__(self, param_dim, poly_degree, n_eta):
        self.param_dim   = param_dim
        self.poly_degree = poly_degree
        self.n_eta       = n_eta

        self.basis_terms = self._generate_polynomial_basis()
        self.num_basis   = len(self.basis_terms)

        self.lyap_basis = [
            cvx.Variable((n_eta, n_eta), symmetric=True)
            for _ in range(self.num_basis)
        ]

    def _generate_polynomial_basis(self):
        """All multivariate monomials in `param_dim` variables up to `poly_degree`."""
        basis = []
        for deg in range(self.poly_degree + 1):
            for term in combinations_with_replacement(range(self.param_dim), deg):
                basis.append(term)
        return basis

    def P(self, p):
        """Construct P(p) as a CVXPY expression."""
        return sum(
            self.lyap_basis[i]
            * np.prod([p[j] ** term.count(j) for j in range(self.param_dim)])
            for i, term in enumerate(self.basis_terms)
        )

    def P_numeric(self, p):
        """Evaluate P(p) numerically using the solved variable values."""
        return sum(
            self.lyap_basis[i].value
            * np.prod([p[j] ** term.count(j) for j in range(self.param_dim)])
            for i, term in enumerate(self.basis_terms)
        )

    def min_max_eigval(self, p_grid):
        """Minimum and maximum eigenvalues of P(p) over a grid of parameter values."""
        min_eig, max_eig = np.inf, -np.inf
        for p in p_grid:
            P_p    = self.P_numeric(p)
            eigvals = np.linalg.eigvalsh(P_p)
            min_eig = min(min_eig, np.min(eigvals))
            max_eig = max(max_eig, np.max(eigvals))
        return min_eig, max_eig

    def condition_P(self, p_grid):
        """Condition number √(λ_max / λ_min) of P(p) over a grid; ∞ if not PD."""
        min_eig, max_eig = self.min_max_eigval(p_grid)
        if min_eig <= 0:
            return np.inf
        return np.sqrt(max_eig / min_eig)
