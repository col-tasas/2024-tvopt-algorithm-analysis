"""
Tests for objectives/definitions.py.

Key properties verified for each objective function:
  - Gradient matches central-difference numerical approximation.
  - Sector bounds m, L are valid (m > 0, L >= m) across a range of times.
  - The returned minimiser x_star satisfies grad f(x_star) ≈ 0.
  - Object dimensions are correct.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_grad(obj, x, eps=1e-6):
    """Central-difference gradient of obj.eval at column vector x."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = float(np.atleast_1d(obj.eval(xp))[0])
        fm = float(np.atleast_1d(obj.eval(xm))[0])
        g[i] = (fp - fm) / (2.0 * eps)
    return g


def _check_sector_bounds(obj, times):
    """Assert m > 0 and L >= m for each t in times."""
    for t in times:
        obj.update(t)
        _, m, L = obj.get_objective_info()
        assert m > 0, f"m must be positive at t={t}, got {m}"
        assert L >= m - 1e-9, f"L ({L}) must be >= m ({m}) at t={t}"


# ---------------------------------------------------------------------------
# PeriodicExample2D
# ---------------------------------------------------------------------------

class TestPeriodicExample2D:
    @pytest.fixture(autouse=True)
    def obj(self):
        from lib.simulation.objectives import PeriodicExample2D
        self._obj = PeriodicExample2D(omega=0.1)
        return self._obj

    def test_nx_is_2(self):
        assert self._obj.nx == 2

    def test_gradient_matches_finite_diff(self):
        self._obj.update(5)
        x = np.array([[1.5], [0.8]])
        g_a = self._obj.gradient(x)
        g_n = _num_grad(self._obj, x)
        np.testing.assert_allclose(g_a, g_n, rtol=1e-4, atol=1e-7)

    def test_sector_bounds_valid(self):
        _check_sector_bounds(self._obj, range(20))

    def test_x_star_is_minimiser(self):
        self._obj.update(10)
        x_star, _, _ = self._obj.get_objective_info()
        g = self._obj.gradient(x_star)
        np.testing.assert_allclose(g, np.zeros_like(g), atol=1e-10)

    def test_hessian_eigenvalues_match_sector_bounds(self):
        """Hessian eigenvalues must lie in [m, L] for all tested times."""
        for t in range(0, 20, 4):
            self._obj.update(t)
            H = self._obj.hessian(np.zeros((2, 1)))
            eigs = np.linalg.eigvalsh(H)
            _, m, L = self._obj.get_objective_info()
            assert eigs.min() >= m - 1e-9
            assert eigs.max() <= L + 1e-9


# ---------------------------------------------------------------------------
# QP (penalised equality)
# ---------------------------------------------------------------------------

class TestQP:
    @pytest.fixture(autouse=True)
    def obj(self):
        from lib.simulation.objectives import QP
        self._obj = QP(rho=1.0)
        return self._obj

    def test_nx_is_2(self):
        assert self._obj.nx == 2

    def test_gradient_matches_finite_diff(self):
        for t in [0, 3, 7]:
            self._obj.update(t)
            x = np.array([[1.0], [-0.5]])
            g_a = self._obj.gradient(x)
            g_n = _num_grad(self._obj, x)
            np.testing.assert_allclose(
                g_a, g_n, rtol=1e-4, atol=1e-7,
                err_msg=f"Gradient mismatch at t={t}",
            )

    def test_sector_bounds_valid(self):
        _check_sector_bounds(self._obj, range(10))

    def test_hessian_eigenvalues_match_sector_bounds(self):
        for t in range(0, 10, 2):
            H = self._obj.hessian(t)
            eigs = np.linalg.eigvalsh(H)
            self._obj.update(t)
            _, m, L = self._obj.get_objective_info()
            assert eigs.min() >= m - 1e-9
            assert eigs.max() <= L + 1e-9


# ---------------------------------------------------------------------------
# QP_unconstrained
# ---------------------------------------------------------------------------

class TestQPUnconstrained:
    @pytest.fixture(autouse=True)
    def obj(self):
        from lib.simulation.objectives import QP_unconstrained
        self._obj = QP_unconstrained()
        return self._obj

    def test_gradient_matches_finite_diff(self):
        self._obj.update(0)
        x = np.array([[2.0], [-1.0]])
        g_a = self._obj.gradient(x)
        g_n = _num_grad(self._obj, x)
        np.testing.assert_allclose(g_a, g_n, rtol=1e-4, atol=1e-7)

    def test_x_star_is_minimiser(self):
        for t in [0, 5, 10]:
            self._obj.update(t)
            x_star, _, _ = self._obj.get_objective_info()
            g = self._obj.gradient(x_star)
            np.testing.assert_allclose(
                g, np.zeros_like(g), atol=1e-8,
                err_msg=f"x_star is not a minimiser at t={t}",
            )

    def test_sector_bounds_valid(self):
        _check_sector_bounds(self._obj, range(10))


# ---------------------------------------------------------------------------
# Robotic_Ellipse_Tracking
# ---------------------------------------------------------------------------

class TestRoboticEllipseTracking:
    @pytest.fixture(autouse=True)
    def obj(self):
        from lib.simulation.objectives import Robotic_Ellipse_Tracking
        self._obj = Robotic_Ellipse_Tracking()
        return self._obj

    def test_nx_is_5(self):
        assert self._obj.nx == 5

    def test_gradient_matches_finite_diff(self):
        self._obj.update(0)
        x = np.zeros((5, 1))
        g_a = self._obj.gradient(x)
        g_n = _num_grad(self._obj, x)
        np.testing.assert_allclose(g_a, g_n, rtol=1e-4, atol=1e-7)

    def test_sector_bounds_positive(self):
        """m and L must be positive (H = I + ρ JᵀJ is SPD)."""
        for t in range(5):
            self._obj.update(t)
            _, m, L = self._obj.get_objective_info()
            assert m > 0
            assert L >= m - 1e-9
