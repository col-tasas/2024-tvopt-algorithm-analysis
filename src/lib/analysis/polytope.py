"""Utilities for generating consistent polytope grids and scanning objective parameter ranges."""

import numpy as np
import matplotlib.pyplot as plt


def consistent_polytope_nd(params, delta_params_min, delta_params_max, step_size=0.1):
    """Generate grid points (p_k, delta_p) satisfying the consistent polytope constraints."""
    if params.ndim == 1:
        params            = params[None, :]
        delta_params_min  = np.array([delta_params_min])
        delta_params_max  = np.array([delta_params_max])

    p_min = np.min(params, axis=1)
    p_max = np.max(params, axis=1)

    grid_points = []

    for k in range(params.shape[1]):
        p_k = params[:, k]

        delta_min_k = np.maximum(delta_params_min,
                                  np.maximum(p_min - p_k, -delta_params_max))
        delta_max_k = np.minimum(delta_params_max,
                                  np.minimum(p_max - p_k, delta_params_max))

        delta_ranges = []
        for d in range(params.shape[0]):
            lo = delta_min_k[d]
            hi = delta_max_k[d]

            if step_size < 1e-8:
                num_pts = 1
            else:
                num_pts = int((hi - lo) / step_size) + 1

            pts = [min(lo + j * step_size, hi) for j in range(num_pts)]
            if pts[-1] < hi:
                pts.append(hi)
            delta_ranges.append(np.array(pts))

        delta_mesh = np.meshgrid(*delta_ranges, indexing='ij')
        delta_combos = np.stack([d.ravel() for d in delta_mesh], axis=-1)

        for delta_p in delta_combos:
            delta_p = np.clip(delta_p, delta_params_min, delta_params_max)
            grid_points.append((p_k, delta_p))

    return grid_points


def visualize(grid_points, param_dim=None):
    """Scatter plots of polytope grid points (debugging aid)."""
    p_values     = np.array([p_k    for p_k, _      in grid_points])
    delta_values = np.array([delta_p for _, delta_p in grid_points])
    num_dims = p_values.shape[1]

    if param_dim is not None:
        if param_dim >= num_dims or param_dim < 0:
            print(f"Invalid dimension for visualization: {param_dim}")
            return
        plt.figure(figsize=(4, 3))
        plt.scatter(p_values[:, param_dim], delta_values[:, param_dim], alpha=0.5)
        plt.xlabel(f"Parameter Values (Dim {param_dim+1})")
        plt.ylabel(f"Delta Values (Dim {param_dim+1})")
        plt.title(f"Grid Point Visualization for Dimension {param_dim+1}")
        plt.show()
    else:
        fig, axes = plt.subplots(num_dims, 1, figsize=(4, 3 * num_dims))
        if num_dims == 1:
            axes = [axes]
        for dim in range(num_dims):
            axes[dim].scatter(p_values[:, dim], delta_values[:, dim], alpha=0.5)
            axes[dim].set_xlabel(f"Parameter Values (Dim {dim+1})")
            axes[dim].set_ylabel(f"Delta Values (Dim {dim+1})")
            axes[dim].set_title(f"Grid Point Visualization for Dimension {dim+1}")
        plt.tight_layout()
        plt.show()
