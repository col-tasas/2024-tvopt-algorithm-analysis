"""
Tests for analysis/polytope.py.

Key properties:
  - consistent_polytope_nd: every returned (p_k, δp) satisfies the four constraints:
      p_min ≤ p_k + δp ≤ p_max   and   δp_min ≤ δp ≤ δp_max
  - Edge case: step_size=0 still returns at least one point.
  - calculate_L_m_bounds: L_min ≤ L_max, m_min ≤ m_max, m > 0, L ≥ m.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# consistent_polytope_nd
# ---------------------------------------------------------------------------

class TestConsistentPolytopeNd:

    def _check_constraints(self, grid_points, params, delta_min, delta_max):
        """Assert every grid point satisfies all four constraints."""
        p_min = np.min(params, axis=1)
        p_max = np.max(params, axis=1)
        tol = 1e-9
        for p_k, delta_p in grid_points:
            p_next = p_k + delta_p
            for d in range(len(p_k)):
                assert p_next[d] >= p_min[d] - tol, (
                    f"Lower bound violated: p_k+δp={p_next[d]:.4f} < p_min={p_min[d]:.4f}"
                )
                assert p_next[d] <= p_max[d] + tol, (
                    f"Upper bound violated: p_k+δp={p_next[d]:.4f} > p_max={p_max[d]:.4f}"
                )
            np.testing.assert_array_less(
                delta_min - tol, delta_p,
                err_msg=f"delta_p {delta_p} below delta_min {delta_min}",
            )
            np.testing.assert_array_less(
                delta_p, delta_max + tol,
                err_msg=f"delta_p {delta_p} above delta_max {delta_max}",
            )

    def test_2d_param_space(self):
        """2D parameter space (m, L) – the main use-case in the library."""
        from lib.analysis.polytope import consistent_polytope_nd
        m_vals = np.array([1.0, 1.5, 2.0])
        L_vals = np.array([3.0, 4.0, 5.0])
        params = np.vstack([m_vals, L_vals])  # (2, 3)

        delta_min = np.array([-0.5, -1.0])
        delta_max = np.array([0.5, 1.0])
        step_size = 0.25

        grid = consistent_polytope_nd(params, delta_min, delta_max, step_size)

        assert len(grid) > 0, "Polytope must not be empty"
        self._check_constraints(grid, params, delta_min, delta_max)

    def test_1d_param_space(self):
        """1D parameter (single L value)."""
        from lib.analysis.polytope import consistent_polytope_nd
        params = np.array([2.0, 3.0, 4.0])  # 1-D input
        grid = consistent_polytope_nd(params, -0.5, 0.5, step_size=0.25)
        assert len(grid) > 0
        for p_k, dp in grid:
            p_min, p_max = float(params.min()), float(params.max())
            val = float(p_k) + float(dp)
            assert val >= p_min - 1e-9
            assert val <= p_max + 1e-9

    def test_step_size_zero_returns_points(self):
        """step_size=0 (or very small) must not crash and must return results."""
        from lib.analysis.polytope import consistent_polytope_nd
        params = np.vstack([np.array([1.0, 2.0]), np.array([3.0, 5.0])])
        delta_min = np.array([-0.2, -0.5])
        delta_max = np.array([0.2, 0.5])
        grid = consistent_polytope_nd(params, delta_min, delta_max, step_size=0.0)
        assert len(grid) > 0

    def test_no_variation_case(self):
        """delta_min = delta_max = 0: only δp=0 is possible."""
        from lib.analysis.polytope import consistent_polytope_nd
        params = np.vstack([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        delta_min = np.array([0.0, 0.0])
        delta_max = np.array([0.0, 0.0])
        grid = consistent_polytope_nd(params, delta_min, delta_max, step_size=0.1)
        assert len(grid) > 0
        for _, dp in grid:
            np.testing.assert_allclose(dp, [0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# calculate_L_m_bounds
# ---------------------------------------------------------------------------

class TestCalculateLmBounds:

    def _assert_valid_bounds(self, L_min, L_max, m_min, m_max):
        assert m_min > 0,             f"m_min must be positive, got {m_min}"
        assert m_max >= m_min - 1e-9, f"m_max ({m_max}) must be >= m_min ({m_min})"
        assert L_min >= m_min - 1e-9, f"L_min ({L_min}) must be >= m_min ({m_min})"
        assert L_max >= L_min - 1e-9, f"L_max ({L_max}) must be >= L_min ({L_min})"

    def test_periodic_2d(self):
        from lib.analysis.polytope import calculate_L_m_bounds
        from lib.simulation.objectives import PeriodicExample2D
        obj = PeriodicExample2D(omega=0.1)
        L_min, L_max, m_min, m_max, dL_max, dm_max, dL_min, dm_min = \
            calculate_L_m_bounds(obj)
        self._assert_valid_bounds(L_min, L_max, m_min, m_max)

    def test_qp(self):
        from lib.analysis.polytope import calculate_L_m_bounds
        from lib.simulation.objectives import QP
        obj = QP(rho=1.0)
        result = calculate_L_m_bounds(obj)
        L_min, L_max, m_min, m_max = result[:4]
        self._assert_valid_bounds(L_min, L_max, m_min, m_max)

    def test_returns_eight_values(self):
        from lib.analysis.polytope import calculate_L_m_bounds
        from lib.simulation.objectives import PeriodicExample2D
        obj = PeriodicExample2D(omega=0.1)
        result = calculate_L_m_bounds(obj)
        assert len(result) == 8, f"Expected 8 return values, got {len(result)}"


# ---------------------------------------------------------------------------
# take_full_sample_of_mL_pairs
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="take_full_sample_of_mL_pairs not yet in lib.analysis.polytope")
class TestTakeFullSample:

    def test_returns_lists_of_correct_length(self):
        from lib.analysis.polytope import take_full_sample_of_mL_pairs
        from lib.simulation.objectives import PeriodicExample2D
        T = 30
        obj = PeriodicExample2D(omega=0.1)
        m_list, L_list = take_full_sample_of_mL_pairs(obj, t=T)
        assert len(m_list) == T + 1
        assert len(L_list) == T + 1

    def test_all_m_positive_and_L_geq_m(self):
        from lib.analysis.polytope import take_full_sample_of_mL_pairs
        from lib.simulation.objectives import PeriodicExample2D
        obj = PeriodicExample2D(omega=0.1)
        m_list, L_list = take_full_sample_of_mL_pairs(obj, t=50)
        for m, L in zip(m_list, L_list):
            assert m > 0
            assert L >= m - 1e-9
