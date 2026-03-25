"""
Tests for algorithm state-space representations and step logic.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Functional API – state-space dimensions
# ---------------------------------------------------------------------------

SINGLE_STEP_ALGOS = [
    ("gradient_descent", 1, 1, 1, 1, 0),
    ("nesterov",         1, 2, 1, 1, 0),
    ("heavy_ball",       1, 2, 1, 1, 0),
    ("triple_momentum",  1, 2, 1, 1, 0),
    ("c2m",              1, 2, 1, 1, 0),
]


@pytest.mark.parametrize("name,K,n_states,n_in,exp_p,exp_q", SINGLE_STEP_ALGOS)
def test_single_step_ss_dimensions(name, K, n_states, n_in, exp_p, exp_q):
    import lib.algorithms.unconstrained as alg
    fn = getattr(alg, name)
    m, L = 1.0, 4.0
    G, p, q = fn(m, L, K=K)
    assert G.nstates == n_states
    assert G.ninputs == n_in
    assert p == exp_p
    assert q == exp_q


@pytest.mark.parametrize("K", [2, 3, 5])
def test_multistep_p_equals_K(K):
    from lib.algorithms.unconstrained import gradient_descent
    G, p, q = gradient_descent(1.0, 4.0, K=K)
    assert p == K
    assert q == 0
    assert G.noutputs == K
    assert G.ninputs == K


@pytest.mark.parametrize("fn_name,K", [
    ("gradient_descent", 3),
    ("nesterov",         3),
    ("heavy_ball",       3),
    ("triple_momentum",  3),
    ("c2m",              3),
])
def test_multistep_D_strictly_lower_triangular(fn_name, K):
    import lib.algorithms.unconstrained as alg
    fn = getattr(alg, fn_name)
    G, p, q = fn(1.0, 4.0, K=K)
    D = G.D
    assert D.shape == (K, K)
    assert np.allclose(np.triu(D), 0)


def test_gradient_descent_step_size():
    from lib.algorithms.unconstrained import gradient_descent
    import control as ctrl
    m, L = 1.0, 3.0
    G, _, _ = gradient_descent(m, L)
    alpha = 2.0 / (m + L)
    A, B, C, D = ctrl.ssdata(G)
    assert float(A) == pytest.approx(1.0)
    assert float(B) == pytest.approx(-alpha, rel=1e-9)
    assert float(C) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

class TestAlgorithmClass:

    def test_gradient_descent_convergence(self):
        from lib.simulation.algorithm import Algorithm
        from lib.algorithms.unconstrained import gradient_descent
        nx = 2
        algo = Algorithm(gradient_descent, nx=nx)
        x0 = np.ones((nx, 1)) * 5.0
        algo.initialize(x0)
        algo.update_sectors(1.0, 1.0)
        for _ in range(60):
            algo.update_gradient(lambda x: x)
            xi, *_ = algo.step()
        assert np.linalg.norm(xi) < 1e-6

    def test_gradient_descent_state_dim(self):
        from lib.simulation.algorithm import Algorithm
        from lib.algorithms.unconstrained import gradient_descent
        for nx in [1, 2, 5]:
            algo = Algorithm(gradient_descent, nx=nx)
            assert algo.internal_state_dim == nx

    def test_nesterov_state_dim_is_2nx(self):
        from lib.simulation.algorithm import Algorithm
        from lib.algorithms.unconstrained import nesterov
        for nx in [1, 2, 5]:
            algo = Algorithm(nesterov, nx=nx)
            assert algo.internal_state_dim == 2 * nx

    def test_nesterov_converges_faster_than_gradient_descent(self):
        from lib.simulation.algorithm import Algorithm
        from lib.algorithms.unconstrained import gradient_descent, nesterov
        m, L, nx = 1.0, 9.0, 1

        def run(algo_fn, n_steps=100):
            algo = Algorithm(algo_fn, nx=nx)
            algo.initialize(np.array([[5.0]]))
            algo.update_sectors(m, L)
            errors = []
            for _ in range(n_steps):
                algo.update_gradient(lambda x: x)
                xi, *_ = algo.step()
                errors.append(float(np.linalg.norm(xi)))
            return errors

        gd_errors = run(gradient_descent)
        nm_errors = run(nesterov)
        assert nm_errors[-1] < gd_errors[-1]

    def test_multistep_D_lower_triangular(self):
        from lib.simulation.algorithm import Algorithm
        from lib.algorithms.unconstrained import gradient_descent
        K, nx = 4, 2
        algo = Algorithm(gradient_descent, nx=nx)
        algo.update_sectors(1.0, 4.0)
        assert np.allclose(np.triu(algo.D), 0)
