"""
Tests for analysis/lure.py (LTI-system building blocks for the IQC framework).

Checks:
  - lti_stack: output dimensions are sum of the two subsystems.
  - build_input_mapping: correct shape for (p, q) without/with vIQC.
  - build_output_mapping: correct shape.
  - build_iqc (static): returns a valid control.ss with expected dimensions.
  - build_lure_system: G_hat has more states than G (IQC filter adds states).
"""

import numpy as np
import pytest
import control as ctrl


# ---------------------------------------------------------------------------
# lti_stack
# ---------------------------------------------------------------------------

class TestLtiStack:

    def test_state_dimension_is_sum(self):
        from lib.analysis.lure import lti_stack
        sys1 = ctrl.ss([0.5], [1.0], [1.0], [0.0], dt=1)  # 1 state
        sys2 = ctrl.ss([[0.3, 0.0], [0.0, 0.2]], [[1.0], [1.0]],
                       [[1.0, 0.0], [0.0, 1.0]], [[0.0], [0.0]], dt=1)  # 2 states
        stacked = lti_stack(sys1, sys2)
        assert stacked.nstates == 3

    def test_output_dimension_is_sum(self):
        from lib.analysis.lure import lti_stack
        sys1 = ctrl.ss([], [], [], np.array([[1.0, 0.0]]), dt=1)
        sys2 = ctrl.ss([], [], [], np.array([[1.0, 0.0], [0.0, 1.0]]), dt=1)
        stacked = lti_stack(sys1, sys2)
        assert stacked.noutputs == 3

    def test_raises_on_input_mismatch(self):
        from lib.analysis.lure import lti_stack
        sys1 = ctrl.ss([], [], [], np.zeros((1, 2)), dt=1)  # 2 inputs
        sys2 = ctrl.ss([], [], [], np.zeros((1, 3)), dt=1)  # 3 inputs
        with pytest.raises(ValueError):
            lti_stack(sys1, sys2)

    def test_block_diagonal_A(self):
        """A of stacked system must be block-diagonal of the two A matrices."""
        from lib.analysis.lure import lti_stack
        A1 = np.array([[0.5]])
        A2 = np.array([[0.3, 0.1], [0.0, 0.2]])
        sys1 = ctrl.ss(A1, [[1.0]], [[1.0]], [[0.0]], dt=1)
        sys2 = ctrl.ss(A2, [[1.0], [1.0]], [[1.0, 0.0]], [[0.0]], dt=1)
        stacked = lti_stack(sys1, sys2)
        A_expected = np.block([[A1, np.zeros((1, 2))], [np.zeros((2, 1)), A2]])
        np.testing.assert_allclose(stacked.A, A_expected)


# ---------------------------------------------------------------------------
# build_input_mapping
# ---------------------------------------------------------------------------

class TestBuildInputMapping:

    def test_shape_without_viqc(self):
        from lib.analysis.lure import build_input_mapping
        p, q = 2, 1
        T = build_input_mapping(p, q, vIQC=False)
        assert T.shape == (2 * (p + q), 2 * (p + q))

    def test_shape_with_viqc(self):
        from lib.analysis.lure import build_input_mapping
        p, q, n_xi = 1, 0, 2
        T = build_input_mapping(p, q, vIQC=True, n_xi=n_xi)
        n_inputs_true    = p + q + p + q + n_xi + p   # = 5
        n_inputs_stacked = p * (3 + n_xi) + q * (2 + n_xi)  # = 5
        assert T.shape == (n_inputs_stacked, n_inputs_true)

    def test_raises_when_n_xi_missing_for_viqc(self):
        from lib.analysis.lure import build_input_mapping
        with pytest.raises(ValueError):
            build_input_mapping(1, 0, vIQC=True)  # missing n_xi


# ---------------------------------------------------------------------------
# build_output_mapping
# ---------------------------------------------------------------------------

class TestBuildOutputMapping:

    def test_square_matrix(self):
        """S must be a square permutation-like matrix."""
        from lib.analysis.lure import build_output_mapping
        p, q = 2, 1
        S = build_output_mapping(p, q)
        total = p * (2 + 6) + q * (2 + 2)
        assert S.shape == (total, total)

    def test_each_row_has_exactly_one_one(self):
        """S is a permutation matrix: each row/col has exactly one non-zero."""
        from lib.analysis.lure import build_output_mapping
        S = build_output_mapping(p=1, q=1)
        row_sums = np.abs(S).sum(axis=1)
        col_sums = np.abs(S).sum(axis=0)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)))
        np.testing.assert_allclose(col_sums, np.ones(len(col_sums)))


# ---------------------------------------------------------------------------
# build_iqc (static)
# ---------------------------------------------------------------------------

class TestBuildIqcStatic:

    @pytest.mark.parametrize("p,q", [(1, 0), (2, 0), (1, 1)])
    def test_returns_ctrl_ss(self, p, q):
        from lib.analysis.lure import build_iqc
        Psi = build_iqc(m=1.0, L=4.0, p=p, q=q, vIQC=False)
        assert isinstance(Psi, ctrl.StateSpace)

    def test_static_iqc_has_no_states(self):
        """Static (non-vIQC) filter must be purely feed-through (0 states)."""
        from lib.analysis.lure import build_iqc
        Psi = build_iqc(m=1.0, L=4.0, p=1, q=0, vIQC=False)
        assert Psi.nstates == 0

    def test_static_iqc_input_count(self):
        """For p=1, q=0, vIQC=False: 2*(p+q) = 2 inputs (one [s, δ] pair)."""
        from lib.analysis.lure import build_iqc
        Psi = build_iqc(m=1.0, L=4.0, p=1, q=0, vIQC=False)
        assert Psi.ninputs == 2


# ---------------------------------------------------------------------------
# build_lure_system
# ---------------------------------------------------------------------------

class TestBuildLureSystem:

    def test_g_hat_nstates_geq_g_nstates(self):
        """Static Lur'e augmentation must produce at least as many states as G."""
        from lib.analysis.lure import build_lure_system
        from lib.algorithms.unconstrained import gradient_descent
        m, L = 1.0, 4.0
        G, p, q = gradient_descent(m, L)
        G_hat, _ = build_lure_system(G, m, L, p, q, vIQC=False)
        assert G_hat.nstates >= G.nstates

    def test_static_lure_returns_ctrl_ss(self):
        from lib.analysis.lure import build_lure_system
        from lib.algorithms.unconstrained import gradient_descent
        m, L = 1.0, 4.0
        G, p, q = gradient_descent(m, L)
        G_hat, Psi = build_lure_system(G, m, L, p, q, vIQC=False)
        assert isinstance(G_hat, ctrl.StateSpace)
        assert isinstance(Psi, ctrl.StateSpace)

    def test_raises_when_rho_missing_for_viqc(self):
        from lib.analysis.lure import build_lure_system
        from lib.algorithms.unconstrained import gradient_descent
        G, p, q = gradient_descent(1.0, 4.0)
        with pytest.raises(ValueError):
            build_lure_system(G, 1.0, 4.0, p, q, vIQC=True)  # missing rho
