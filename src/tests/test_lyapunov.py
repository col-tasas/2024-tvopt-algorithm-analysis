"""
Tests for analysis/lyapunov.py – PolynomialLyapunovMatrix.

Key properties:
  - Polynomial basis has the correct cardinality.
  - P(p) returns a valid CVXPY expression.
  - condition_P returns inf when the basis matrices are zero (not PD).
  - After a tiny SDP solve, P_numeric and min_max_eigval work correctly.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Basis generation
# ---------------------------------------------------------------------------

class TestPolynomialBasis:
    """
    Number of monomials in d variables up to degree k is C(d+k, k).
    """

    @pytest.mark.parametrize("param_dim,poly_degree,expected", [
        (1, 0, 1),
        (1, 1, 2),
        (1, 2, 3),
        (2, 0, 1),
        (2, 1, 3),
        (2, 2, 6),
    ])
    def test_basis_cardinality(self, param_dim, poly_degree, expected):
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        plm = PolynomialLyapunovMatrix(param_dim=param_dim, poly_degree=poly_degree, n_eta=2)
        assert len(plm.basis_terms) == expected, (
            f"param_dim={param_dim}, degree={poly_degree}: "
            f"expected {expected} terms, got {len(plm.basis_terms)}"
        )

    def test_degree_0_has_constant_term_only(self):
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        plm = PolynomialLyapunovMatrix(param_dim=2, poly_degree=0, n_eta=3)
        assert plm.basis_terms == [()]


# ---------------------------------------------------------------------------
# P(p) – CVXPY expression
# ---------------------------------------------------------------------------

class TestLyapunovExpression:

    def test_P_is_cvxpy_expression(self):
        import cvxpy as cvx
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        plm = PolynomialLyapunovMatrix(param_dim=1, poly_degree=1, n_eta=2)
        p = np.array([2.0])
        P_expr = plm.P(p)
        assert isinstance(P_expr, cvx.Expression), \
               "P(p) should return a CVXPY Expression"

    def test_P_has_correct_shape(self):
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        n_eta = 3
        plm = PolynomialLyapunovMatrix(param_dim=2, poly_degree=1, n_eta=n_eta)
        p = np.array([1.5, 3.0])
        P_expr = plm.P(p)
        assert P_expr.shape == (n_eta, n_eta)

    def test_num_basis_variables_created(self):
        """One SDP variable per basis term, each of shape (n_eta, n_eta)."""
        import cvxpy as cvx
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        n_eta, param_dim, degree = 2, 1, 2
        plm = PolynomialLyapunovMatrix(param_dim=param_dim, poly_degree=degree, n_eta=n_eta)
        assert len(plm.lyap_basis) == len(plm.basis_terms)
        for var in plm.lyap_basis:
            assert isinstance(var, cvx.Variable)
            assert var.shape == (n_eta, n_eta)


# ---------------------------------------------------------------------------
# condition_P – returns inf when not PD
# ---------------------------------------------------------------------------

class TestConditionP:

    def test_returns_inf_when_basis_not_set(self):
        """Freshly constructed PLM has zero basis matrices -> not PD -> inf."""
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        plm = PolynomialLyapunovMatrix(param_dim=1, poly_degree=0, n_eta=2)
        for var in plm.lyap_basis:
            var._value = np.zeros((2, 2))
        p_grid = [np.array([1.0])]
        cond = plm.condition_P(p_grid)
        assert cond == np.inf

    def test_condition_after_manual_assignment(self):
        """If the single basis matrix is the identity, condition number is 1."""
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        n_eta = 3
        plm = PolynomialLyapunovMatrix(param_dim=1, poly_degree=0, n_eta=n_eta)
        plm.lyap_basis[0]._value = np.eye(n_eta)
        p_grid = [np.array([1.0]), np.array([2.0])]
        cond = plm.condition_P(p_grid)
        assert cond == pytest.approx(1.0, rel=1e-6)

    def test_min_max_eigval(self):
        """min/max eigenvalue over a grid."""
        from lib.analysis.lyapunov import PolynomialLyapunovMatrix
        n_eta = 2
        plm = PolynomialLyapunovMatrix(param_dim=1, poly_degree=0, n_eta=n_eta)
        P_val = np.array([[4.0, 0.0], [0.0, 1.0]])
        plm.lyap_basis[0]._value = P_val
        p_grid = [np.array([1.0])]
        lo, hi = plm.min_max_eigval(p_grid)
        assert lo == pytest.approx(1.0, rel=1e-9)
        assert hi == pytest.approx(4.0, rel=1e-9)
