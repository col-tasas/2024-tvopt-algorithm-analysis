"""Simulation class for optimization algorithms in the Lure state-space framework."""

import control as ctrl
import numpy as np


class Algorithm:
    """Simulates one optimization algorithm in the Lure framework."""

    def __init__(self, algo_realization, nx=1):
        self.algo_realization = algo_realization
        self.nx = nx
        self.internal_state = None
        self.gradient_function = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.p = None
        self.q = None
        self.internal_state_dim = None
        self._update_state_space(1, 1)

    def _update_state_space(self, m, L):
        G, p, q = self.algo_realization(m, L)
        self.A, self.B, self.C, self.D = self._kron_expand(G)
        self.p = p
        self.q = q
        self.internal_state_dim = self.A.shape[0]
        self.m = m
        self.L = L

    def initialize(self, x0):
        """Set internal state, tiling x0 to fill the full state dimension."""
        n_tiles = self.internal_state_dim // self.nx
        self.internal_state = np.tile(x0.reshape(self.nx, 1), (n_tiles, 1)).reshape(-1, 1)

    def update_sectors(self, m, L):
        self._update_state_space(m, L)

    def update_gradient(self, fn):
        self.gradient_function = fn

    def _compute_sequential_part(self, y_k):
        """Evaluate strictly-lower-triangular D feed-through sequentially."""
        g_k = np.zeros((y_k.shape[0], 1))
        for i in range(y_k.shape[0] // self.nx):
            for j in range(self.nx):
                if i == 0:
                    if not np.all(np.triu(self.D, -self.nx + 1) == 0):
                        break
                else:
                    y_k[i * self.nx + j] += self.D[i * self.nx + j] @ g_k
                    
            g_k[i * self.nx:(i + 1) * self.nx] = self.gradient_function(y_k[i * self.nx:(i + 1) * self.nx])
        return y_k

    def step(self):
        """Advance one time step; returns (xi_k, x_k, y_k)."""
        if self.internal_state is None:
            raise ValueError("Call initialize() before step().")
        if self.gradient_function is None:
            raise ValueError("Call update_gradient() before step().")

        y_k = self.C @ self.internal_state

        if np.any(self.D):
            y_k = self._compute_sequential_part(y_k)

        g_k = self.gradient_function(y_k)
        self.internal_state = self.A @ self.internal_state + self.B @ g_k

        # For multi-step algorithms y_k is stacked; return only first nx elements
        x_k = y_k[:self.nx]
        return self.internal_state, x_k, y_k


    def get_state_space(self):
        return ctrl.ss(self.A, self.B, self.C, self.D, dt=1)


    def _kron_expand(self, G):
        """Extract (A, B, C, D) from a scalar ctrl.StateSpace and Kronecker-expand to nx."""
        A, B, C, D = ctrl.ssdata(G)
        I = np.eye(self.nx)
        return (np.kron(A, I), np.kron(B, I), np.kron(C, I), np.kron(D, I))