import cvxpy as cvx
import numpy as np
from itertools import combinations_with_replacement


class PolynomialLyapunovMatrix:
    def __init__(self, param_dim, poly_degree, n_eta):
        self.param_dim = param_dim      # Dimension of p
        self.poly_degree = poly_degree  # Degree of polynomial
        self.n_eta = n_eta              # Dimension of matrix
        
        # Generate polynomial basis terms
        self.basis_terms = self.generate_polynomial_basis()
        self.num_basis = len(self.basis_terms)
        
        # Create SDP variables for each polynomial basis term
        self.lyap_basis = [cvx.Variable((n_eta, n_eta), symmetric=True) for _ in range(self.num_basis)]
    

    def generate_polynomial_basis(self):
        """Generates polynomial basis terms up to the specified degree."""
        basis_terms = []
        for deg in range(self.poly_degree + 1):
            for term in combinations_with_replacement(range(self.param_dim), deg):
                basis_terms.append(term)
        return basis_terms
    

    def P(self, p):
        """Constructs the polynomial Lyapunov matrix as a cvxpy expression."""
        P_p = sum(self.lyap_basis[i] * np.prod([p[j]**term.count(j) for j in range(self.param_dim)])
                   for i, term in enumerate(self.basis_terms))
        return P_p
    

    def P_numeric(self, p):
        """Evaluates P(p) numerically given the values of the SDP variables."""
        P_p = sum(self.lyap_basis[i].value * np.prod([p[j]**term.count(j) for j in range(self.param_dim)])
                   for i, term in enumerate(self.basis_terms))
        return P_p
    

    def min_max_eigval(self, p_grid):
        """Computes the minimum and maximum eigenvalues of P(p) over a grid."""
        min_eig, max_eig = np.inf, -np.inf
        
        for p in p_grid:
            P_p = self.P_numeric(p)
            eigvals = np.linalg.eigvalsh(P_p)
            
            min_eig = min(min_eig, np.min(eigvals))
            max_eig = max(max_eig, np.max(eigvals))
        
        return min_eig, max_eig
    
    
    def condition_P(self, p_grid):
        """Computes the condition number of P(p) over the grid."""
        min_eig, max_eig = self.min_max_eigval(p_grid)
        
        if min_eig <= 0:
            return np.inf  # P(p) is not positive definite
        
        return np.sqrt(max_eig / min_eig)