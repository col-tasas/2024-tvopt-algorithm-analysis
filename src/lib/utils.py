import numpy as np

def consistent_polytope_nd(params, delta_params_min, delta_params_max, step_size=0.1):
    """
    Generates grid points such that p_min <= p + delta_p <= p_max
    and delta_p_min <= delta_p <= delta_p_max for each parameter vector.

    Parameters:
        params (numpy.ndarray): A 2D array where each row represents a parameter vector.
        delta_params_min (numpy.ndarray): Minimum allowable deltas for each parameter.
        delta_params_max (numpy.ndarray): Maximum allowable deltas for each parameter.
        step_size (float): Step size for generating grid points in the delta ranges.

    Returns:
        list: A list of tuples, each containing (p_k, delta_p) where p_k is the parameter vector
              and delta_p is the corresponding delta vector satisfying the constraints.
    """
    # Handle 1D input by converting it to 2D for uniform processing
    if params.ndim == 1:
        params = params[None, :]  # Convert to shape (1, n)
        delta_params_min = np.array([delta_params_min])
        delta_params_max = np.array([delta_params_max])

    # Calculate global min and max for params directly using numpy
    p_min = np.min(params, axis=1)
    p_max = np.max(params, axis=1)

    # Initialize the grid points for (p, delta_p)
    grid_points = []

    # Loop through each parameter vector
    for k in range(params.shape[1]):
        p_k = params[:, k]

        # Determine feasible ranges for deltas for each dimension
        delta_min_k = np.maximum(delta_params_min, np.maximum(p_min - p_k, -delta_params_max))
        delta_max_k = np.minimum(delta_params_max, np.minimum(p_max - p_k, delta_params_max))

        # Generate grid points for all dimensions
        delta_ranges = []
        for d in range(params.shape[0]):
            delta_min = delta_min_k[d]
            delta_max = delta_max_k[d]
            num_dp_points = int((delta_max - delta_min) / step_size) + 1
            range_points = [
                min(delta_min + j * step_size, delta_max)  # Ensure boundary inclusion
                for j in range(num_dp_points)
            ]
            if range_points[-1] < delta_max:
                range_points.append(delta_max)

            delta_ranges.append(np.array(range_points))

        # Create a meshgrid of delta ranges and iterate through combinations
        delta_mesh = np.meshgrid(*delta_ranges, indexing="ij")
        delta_combinations = np.stack([delta.ravel() for delta in delta_mesh], axis=-1)

        for delta_p in delta_combinations:
            delta_p = np.clip(delta_p, delta_params_min, delta_params_max)
            grid_points.append((p_k, delta_p))

    return grid_points