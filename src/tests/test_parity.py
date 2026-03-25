"""
Output-parity regression tests.

Verifies that the current implementation reproduces the numerical values that
were established as correct during the March 2026 refactoring from src/ to the
lib package.  Two categories of checks:

SimParity
---------
Re-runs ``simulate_once`` with the exact inputs stored in the sim-cache sidecar
JSON and asserts that the output matches the cached (tracking_error, error_bound)
pair within tight tolerances (rtol=1e-6).  The cached values were written by code
that was verified bit-for-bit identical to the pre-refactoring src/ implementation.
No SDP solver is required.

SensParity
----------
Runs ``bisection_static_iqc`` on a fixed small polytope
(PeriodicExample2D + gradient descent, 8 polytope points) and asserts that the
results match the values confirmed against the pre-refactoring src/ implementation:

    bisection_static_iqc  -> rho ~ 0.35645,  cond_P ~ 2.1015

All SensParity tests are marked ``slow`` because they require the MOSEK solver.
"""

import glob
import json
import os
import pickle

import numpy as np
import pytest

_TESTS_DIR    = os.path.dirname(__file__)
# Fixture cache: a small curated subset of the full repo cache, checked into
# tests/fixtures/sim_cache/ so the tests have no external dependency.
_FIXTURE_ROOT = os.path.join(_TESTS_DIR, 'fixtures', 'sim_cache')


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sim_cache_files(obj_name, algo_name):
    """Return all sim-cache pkl paths for a given objective/algo from the fixtures."""
    pattern = os.path.join(
        _FIXTURE_ROOT, f'obj_{obj_name}', 'sim_cache', algo_name, '*.pkl'
    )
    return sorted(glob.glob(pattern))


def _load_sim_case(pkl_path):
    """
    Load a cached simulation result and reconstruct the inputs from the
    accompanying sidecar JSON.

    The sidecar is named ``{hash}.json`` in the same directory as the pkl,
    where *hash* is the last ``_``-separated segment of the pkl stem.

    Returns
    -------
    (te_cached, eb_cached, algo_name, T, x0, cert) or None if the sidecar
    is missing or the certificate structure is not recognised.
    """
    stem     = os.path.splitext(os.path.basename(pkl_path))[0]
    h        = stem.rsplit('_', 1)[-1]        # e.g. "022083db72a2"
    sidecar  = os.path.join(os.path.dirname(pkl_path), f'{h}.json')

    if not os.path.exists(sidecar):
        return None

    with open(pkl_path, 'rb') as f:
        te_cached, eb_cached = pickle.load(f)

    with open(sidecar) as f:
        meta = json.load(f)

    s = meta['sens']
    if 'c1' in s:
        # Old format: sensitivity_f was a scalar, lambdas was a list.
        # New format: lambda_f is a tuple (the lambdas).
        lambdas = s.get('lambdas', [s.get('sensitivity_f', 0.0)])
        cert = {
            'iqcType': 'variational',
            'rho': s['rho'], 'c1': s['c1'], 'c2': s['c2'],
            'lambda_f': tuple(lambdas),
            'sensitivity_x': s['sensitivity_x'],
            'sensitivity_g': s['sensitivity_g'],
        }
    elif 'c' in s:
        cert = {'iqcType': 'static', 'rho': s['rho'], 'c': s['c']}
    else:
        return None

    x0 = np.array(meta['x0']).reshape(-1, 1)
    return te_cached, eb_cached, meta['algo'], meta['T'], x0, cert


# ---------------------------------------------------------------------------
# SimParity: re-run simulate_once, expect output within rtol=1e-6 of cache
# ---------------------------------------------------------------------------

_ALGO_INIT = {
    'gradient':  lambda: __import__('lib.algorithms.unconstrained',
                                    fromlist=['gradient_descent']).gradient_descent,
    'nesterov':  lambda: __import__('lib.algorithms.unconstrained',
                                    fromlist=['nesterov']).nesterov,
    'tmm':       lambda: __import__('lib.algorithms.unconstrained',
                                    fromlist=['triple_momentum']).triple_momentum,
    'heavy_ball': lambda: __import__('lib.algorithms.unconstrained',
                                     fromlist=['heavy_ball']).heavy_ball,
    'c2m':       lambda: __import__('lib.algorithms.unconstrained',
                                    fromlist=['c2m']).c2m,
}


def _get_algo_fn(algo_name):
    from lib.algorithms import unconstrained as alg
    _map = {
        'gradient':   alg.gradient_descent,
        'nesterov':   alg.nesterov,
        'tmm':        alg.triple_momentum,
        'heavy_ball': alg.heavy_ball,
        'c2m':        alg.c2m,
    }
    return _map[algo_name]


class TestSimParity:
    """
    Re-run ``simulate_once`` with the same inputs stored in the sidecar JSON
    and assert that the result matches the cached (tracking_error, error_bound) pair.

    The tolerance rtol=1e-6 is tight enough to catch algorithmic regressions
    while allowing for sub-machine-epsilon floating-point differences that
    arise from np.float64 vs Python float accumulation.
    """

    RTOL = 1e-6   # relative tolerance for element-wise comparison
    ATOL = 1e-12  # absolute tolerance (guards against zero-denominator issues)

    def _check_parity(self, obj_name, algo):
        files = _sim_cache_files(obj_name, algo)
        # Fixture files must always be present – fail loudly if they are missing.
        assert files, (
            f'Fixture missing: tests/fixtures/sim_cache/obj_{obj_name}/sim_cache/{algo}/'
            f' contains no .pkl files.  Re-populate with the copy script.'
        )

        from lib.simulation.simulate import simulate
        from lib.simulation.objectives import PeriodicExample2D, QP

        _OBJ_INIT = {
            'periodic_example_2D': lambda: PeriodicExample2D(omega=0.1),
            'qp': lambda: QP(),
        }

        for pkl_path in files:
            case = _load_sim_case(pkl_path)
            assert case is not None, (
                f'Sidecar JSON missing for fixture {os.path.basename(pkl_path)}.'
            )

            te_cached, eb_cached, algo_name, T, x0, cert = case

            obj = _OBJ_INIT[obj_name]()
            algo_fn = _get_algo_fn(algo_name)
            tracking_error_new, error_bound_new = simulate(
                algo_fn, obj, x0, T, cert
            )

            np.testing.assert_allclose(
                tracking_error_new, te_cached,
                rtol=self.RTOL, atol=self.ATOL,
                err_msg=(
                    f'Tracking-error mismatch for {obj_name}/{algo}: '
                    f'{os.path.basename(pkl_path)}'
                ),
            )
            np.testing.assert_allclose(
                error_bound_new, eb_cached,
                rtol=self.RTOL, atol=self.ATOL,
                err_msg=(
                    f'Error-bound mismatch for {obj_name}/{algo}: '
                    f'{os.path.basename(pkl_path)}'
                ),
            )

    @pytest.mark.parametrize('obj_name,algo', [
        ('periodic_example_2D', 'gradient'),
        ('periodic_example_2D', 'nesterov'),
        ('qp',                  'gradient'),
        ('qp',                  'nesterov'),
    ])
    def test_simulate_once_output_matches_cache(self, obj_name, algo):
        """simulate_once must reproduce the cached tracking error and bound."""
        self._check_parity(obj_name, algo)

    def test_simulate_once_is_deterministic(self):
        """
        Running simulate_once twice with the same inputs must produce
        bit-for-bit identical results (no hidden randomness).
        """
        from lib.simulation.objectives import PeriodicExample2D
        from lib.simulation.simulate import simulate
        from lib.algorithms.unconstrained import gradient_descent

        obj1 = PeriodicExample2D(omega=0.1)
        obj2 = PeriodicExample2D(omega=0.1)
        x0   = np.ones((2, 1)) * 5.0
        T    = 30
        cert = {
            'iqcType': 'variational',
            'rho': 0.75, 'c1': 50.0, 'c2': 20.0,
            'lambda_f': (0.05,), 'sensitivity_x': 5.0, 'sensitivity_g': 3.0,
        }

        tracking_error1, error_bound1 = simulate(gradient_descent, obj1, x0, T, cert)
        tracking_error2, error_bound2 = simulate(gradient_descent, obj2, x0, T, cert)

        assert tracking_error1 == tracking_error2, 'simulate_once is not deterministic (tracking error differs)'
        assert error_bound1 == error_bound2, 'simulate_once is not deterministic (error bound differs)'


# ---------------------------------------------------------------------------
# SensParity: run bisection, expect values matching refactoring verification
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def _periodic_polytope():
    """
    Build the small PeriodicExample2D polytope used for SensParity checks.

    Identical setup to the March 2026 parity verification:
    ``step_size=1.0`` -> 8 polytope points.
    """
    from lib.simulation.objectives  import PeriodicExample2D
    from lib.analysis.polytope       import (
        calculate_L_m_bounds, consistent_polytope_nd,
    )

    obj = PeriodicExample2D(omega=0.1)
    L_min, L_max, m_min, m_max, dL_max, dm_max, dL_min, dm_min = \
        calculate_L_m_bounds(obj)

    params    = np.array([[m_min, m_max], [L_min, L_max]])
    delta_min = np.array([dm_min, dL_min])
    delta_max = np.array([dm_max, dL_max])
    return consistent_polytope_nd(params, delta_min, delta_max, step_size=1.0)


class TestSensParity:
    """
    Runs the SDP bisection on a fixed small polytope and asserts the results
    match the values established against the pre-refactoring src/ implementation.

    Known-good values (from March 2026 parity run, diff=0.00e+00 vs old code):
        bisection_static_iqc: rho=0.356445, cond_P=2.101533

    All tests are marked ``slow`` because they invoke the MOSEK solver.
    """

    # rho bisection tolerance is 1e-3 -> 2e-3 is the right assertion bound.
    TOL_RHO = 2e-3
    # SDP solutions can vary slightly; use 1 % relative tolerance for the rest.
    TOL_REL = 0.01

    @staticmethod
    def _run_static_iqc(polytope):
        from lib.algorithms.unconstrained import gradient_descent
        from lib.analysis.run_solver      import static_IQC_rho_bisection
        cert = static_IQC_rho_bisection(
            gradient_descent, polytope,
            rho_max=1.0, eps=1e-4,
        )
        return (cert['rho'], cert['c'])

    # ------------------------------------------------------------------
    # Static IQC (Theorem 5.2) bisection
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_static_iqc_rho(self, _periodic_polytope):
        """bisection_static_iqc must return rho ~ 0.35645 on the fixed polytope."""
        rho, _ = self._run_static_iqc(_periodic_polytope)
        assert abs(rho - 0.356445) < self.TOL_RHO, (
            f'static_iqc rho={rho:.6f}, expected ~0.356445 (tol={self.TOL_RHO})'
        )

    @pytest.mark.slow
    def test_static_iqc_cond_P_positive(self, _periodic_polytope):
        """bisection_static_iqc must return a positive cond_P (condition number of P)."""
        _, cond_P = self._run_static_iqc(_periodic_polytope)
        assert cond_P is not None, 'bisection_static_iqc returned no feasible solution'
        assert cond_P > 0, f'cond_P must be positive, got {cond_P}'

    @pytest.mark.slow
    def test_static_iqc_cond_P_value(self, _periodic_polytope):
        """bisection_static_iqc cond_P must match the verified value ~ 2.1015."""
        EXPECTED = 2.101533
        _, cond_P = self._run_static_iqc(_periodic_polytope)
        assert abs(cond_P - EXPECTED) < max(self.TOL_RHO, EXPECTED * self.TOL_REL), (
            f'static_iqc cond_P={cond_P:.6f}, expected ~{EXPECTED}'
        )
