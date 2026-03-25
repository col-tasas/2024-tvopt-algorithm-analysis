"""Plot helpers for tracking error and IQC bound visualisation."""

import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_bounds_figure(
    tracking_error,
    sIQC_bounds=None,
    vIQC_bounds=None,
    algo_name='',
    save_plots=False,
    plots_path=None,
):
    """Plot tracking error alongside Static and Varying IQC certified bounds on a semilog axis."""
    plt.figure(figsize=(14, 7))
    h_track = plt.semilogy(tracking_error, label=f"{algo_name} tracking error")[0]

    for idx, (_, error_bound, cert) in enumerate(sIQC_bounds):
        a = 0.9 * np.exp(-idx / 4.0) + 0.1
        plt.semilogy(error_bound, linestyle='-.', alpha=a,
                     label=f"Static IQC #{idx+1}: rho={cert['rho']:.3g}")

    for idx, (_, error_bound, cert) in enumerate(vIQC_bounds):
        a = 0.9 * np.exp(-idx / 4.0) + 0.1
        plt.semilogy(error_bound, linestyle='--', alpha=a,
                     label=f"Variational IQC #{idx+1}: rho={cert['rho']:.3g}")

    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(fontsize='small', frameon=True)
    plt.title(f"{algo_name}: tracking error and bounds")
    plt.tight_layout()

    if save_plots:
        os.makedirs(plots_path or '.', exist_ok=True)
        ts    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"bounds_{algo_name}_{ts}.pdf"
        plt.savefig(os.path.join(plots_path or '.', fname))
    else:
        plt.show()
