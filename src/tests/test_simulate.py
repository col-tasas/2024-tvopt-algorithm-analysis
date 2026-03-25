"""
Simulation tests – verifying simulate_once output shapes, signs, and cached gold standards.
"""

import glob
import os
import pickle

import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(__file__)
_CACHE_ROOT = os.path.join(_TESTS_DIR, "fixtures", "sim_cache")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sim_cache_files(obj_name, algo_name, max_files=5):
    pattern = os.path.join(_CACHE_ROOT, f"obj_{obj_name}", "sim_cache",
                           algo_name, "*.pkl")
    return glob.glob(pattern)[:max_files]


def _make_cert(rho=0.75, c1=10.0, c2=5.0,
               sensitivity_f=(0.1,), sensitivity_x=2.0, sensitivity_g=1.5):
    return {
        'iqcType': 'variational',
        'rho': rho, 'c1': c1, 'c2': c2,
        'lambda_f': sensitivity_f,
        'sensitivity_x': sensitivity_x,
        'sensitivity_g': sensitivity_g,
    }


def _make_sector_cert(rho=0.75, c=3.0):
    return {'iqcType': 'static', 'rho': rho, 'c': c}


# ---------------------------------------------------------------------------
# simulate_once – shape and sign checks
# ---------------------------------------------------------------------------

class TestSimulateOnce:

    @pytest.fixture
    def periodic_obj(self):
        from lib.simulation.objectives import PeriodicExample2D
        return PeriodicExample2D(omega=0.1)

    def test_viqc_output_shapes(self, periodic_obj):
        from lib.simulation.simulate import simulate
        from lib.algorithms.unconstrained import gradient_descent
        T = 30
        x0 = np.ones((2, 1)) * 5.0
        cert = _make_cert(rho=0.75, c1=50.0, c2=20.0,
                          sensitivity_f=(0.05,), sensitivity_x=5.0, sensitivity_g=3.0)
        tracking_error, error_bound = simulate(gradient_descent, periodic_obj, x0, T, cert)
        assert len(tracking_error) == T
        assert len(error_bound) == T

    def test_viqc_bounds_non_negative(self, periodic_obj):
        from lib.simulation.simulate import simulate
        from lib.algorithms.unconstrained import gradient_descent
        T = 20
        x0 = np.ones((2, 1)) * 5.0
        cert = _make_cert(rho=0.80, c1=100.0, c2=30.0,
                          sensitivity_f=(0.1,), sensitivity_x=3.0, sensitivity_g=2.0)
        tracking_error, error_bound = simulate(gradient_descent, periodic_obj, x0, T, cert)
        assert all(e >= -1e-10 for e in tracking_error)
        assert all(b >= -1e-10 for b in error_bound)

    def test_sector_output_shapes(self, periodic_obj):
        from lib.simulation.simulate import simulate
        from lib.algorithms.unconstrained import gradient_descent
        T = 20
        x0 = np.ones((2, 1)) * 5.0
        cert = _make_sector_cert(rho=0.75, c=5.0)
        tracking_error, error_bound = simulate(gradient_descent, periodic_obj, x0, T, cert)
        assert len(tracking_error) == T
        assert len(error_bound) == T
        assert all(b >= -1e-10 for b in error_bound)

    def test_first_tracking_error_zero(self, periodic_obj):
        from lib.simulation.simulate import simulate
        from lib.algorithms.unconstrained import gradient_descent
        periodic_obj.update(0)
        x_star, _, _ = periodic_obj.get_objective_info()
        cert = _make_cert()
        tracking_error, _ = simulate(gradient_descent, periodic_obj, x_star, 10, cert)
        assert tracking_error[0] == pytest.approx(0.0, abs=1e-12)

    def test_is_deterministic(self, periodic_obj):
        from lib.simulation.simulate import simulate
        from lib.algorithms.unconstrained import gradient_descent
        from lib.simulation.objectives import PeriodicExample2D
        obj1 = PeriodicExample2D(omega=0.1)
        obj2 = PeriodicExample2D(omega=0.1)
        x0 = np.ones((2, 1)) * 5.0
        cert = _make_cert()
        tracking_error1, error_bound1 = simulate(gradient_descent, obj1, x0, 30, cert)
        tracking_error2, error_bound2 = simulate(gradient_descent, obj2, x0, 30, cert)
        assert tracking_error1 == tracking_error2
        assert error_bound1 == error_bound2


# ---------------------------------------------------------------------------
# Gold-standard: cached (tracking_error, error_bound) pairs
# ---------------------------------------------------------------------------

class TestCachedBoundsGoldStandard:
    def _assert_bound_property(self, tracking_error, error_bound):
        te = np.asarray(tracking_error, dtype=float)
        eb = np.asarray(error_bound, dtype=float)
        violations = np.where(eb < te - 1e-10)[0]
        assert len(violations) == 0

    @pytest.mark.parametrize("algo", ["gradient", "nesterov"])
    def test_periodic_2d_offby1_bounds(self, algo):
        files = _sim_cache_files("periodic_example_2D", algo, max_files=10)
        if not files:
            pytest.skip(f"No sim cache for periodic_example_2D/{algo}")
        for path in files:
            with open(path, 'rb') as f:
                tracking_error, error_bound = pickle.load(f)
            if error_bound:
                self._assert_bound_property(tracking_error, error_bound)
