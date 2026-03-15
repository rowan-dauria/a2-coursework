"""Plotting and visualisation utilities for CT reconstruction."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_sinogram_grid(sino_dict, row_params, I0_levels, row_label="angles",
                       title="Noisy Sinograms"):
    """Plot a grid of sinograms (rows = scan parameter, cols = I0 levels).

    Parameters
    ----------
    sino_dict : dict
        Nested dict ``sino_dict[row_param][I0] -> sinogram``.
    row_params : Sequence
        Values for the row parameter (e.g. angle counts or angular ranges).
    I0_levels : Sequence[float]
        Source intensity levels for columns.
    row_label : str
        Label describing the row parameter (e.g. ``"angles"`` or ``"° range"``).
    title : str
        Figure super-title.
    """
    n_rows = len(row_params)
    n_cols = len(I0_levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 14))

    # Handle single-row case
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, param in enumerate(row_params):
        for j, I0 in enumerate(I0_levels):
            ax = axes[i, j]
            im = ax.imshow(sino_dict[param][I0], cmap="gray", aspect="auto")
            ax.set_title(f"{param} {row_label}, $I_0$ = {I0:.0e}")
            ax.set_xlabel("Projection angle index")
            ax.set_ylabel("Detector position")
            fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_comparison(results, row_params, I0_levels,
                                   row_label="angles", title="FBP vs GD"):
    """Plot side-by-side FBP and GD reconstructions with metrics.

    Parameters
    ----------
    results : dict
        Nested dict ``results[row_param][I0]["fbp"/"gd"]["image"/"metrics"]``.
    row_params : Sequence
        Values for the row parameter.
    I0_levels : Sequence[float]
        Source intensity levels.
    row_label : str
        Label describing the row parameter.
    title : str
        Figure super-title.
    """
    n_rows = len(row_params)
    n_cols = len(I0_levels) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 14))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, param in enumerate(row_params):
        for j, I0 in enumerate(I0_levels):
            res = results[param][I0]

            for k, method in enumerate(["fbp", "gd"]):
                ax = axes[i, j * 2 + k]
                ax.imshow(res[method]["image"], cmap="gray")
                m = res[method]["metrics"]
                ax.set_title(
                    f"{method.upper()} | {param} {row_label}, "
                    f"$I_0$={I0:.0e}\n"
                    f"PSNR={m['PSNR']:.1f}, SSIM={m['SSIM']:.3f}",
                    fontsize=9,
                )
                ax.axis("off")

    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def print_metrics_table(results, row_params, I0_levels, row_label="Angles"):
    """Print a summary table of reconstruction metrics.

    Parameters
    ----------
    results : dict
        Nested dict ``results[row_param][I0]["fbp"/"gd"]["metrics"]``.
    row_params : Sequence
        Values for the row parameter.
    I0_levels : Sequence[float]
        Source intensity levels.
    row_label : str
        Column header for the row parameter.
    """
    print(f"{row_label:>6} | {'I0':>8} | {'Method':>4} | "
          f"{'RMSE':>8} | {'PSNR':>8} | {'SSIM':>8}")
    print("-" * 60)

    for param in row_params:
        for I0 in I0_levels:
            for method in ["fbp", "gd"]:
                m = results[param][I0][method]["metrics"]
                print(
                    f"{param:>6} | {I0:>8.0e} | {method.upper():>4} | "
                    f"{m['RMSE']:>8.4f} | {m['PSNR']:>8.2f} | "
                    f"{m['SSIM']:>8.4f}"
                )
