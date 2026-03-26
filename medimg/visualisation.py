"""Plotting and visualisation utilities for CT reconstruction."""

import matplotlib.pyplot as plt
import numpy as np


def plot_sinogram_grid(
    sino_dict, row_params, I0_levels, row_label="angles", title="Noisy Sinograms"
):
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


def plot_reconstruction_comparison(
    results, row_params, I0_levels, row_label="angles", title="FBP vs GD"
):
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
    print(
        f"{row_label:>6} | {'I0':>8} | {'Method':>4} | "
        f"{'RMSE':>8} | {'PSNR':>8} | {'SSIM':>8}"
    )
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


def plot_coil_grid(
    images, n_coils, cmap="gray", log_scale=False, title="", label="Coil"
):
    """Plot a grid of per-coil 2D images.

    Parameters
    ----------
    images : np.ndarray
        Array of shape ``(n_coils, H, W)``.
    n_coils : int
        Number of coils to plot.
    cmap : str
        Matplotlib colourmap name.
    log_scale : bool
        If True, apply ``np.log1p(np.abs(...))`` before display.
    title : str
        Figure super-title.
    label : str
        Label prefix for each subplot title.
    """
    ncols = min(n_coils, 3)
    nrows = int(np.ceil(n_coils / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(n_coils):
        ax = axes[i // ncols, i % ncols]
        data = np.abs(images[i])
        if log_scale:
            data = np.log1p(data)
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(f"{label} {i}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused axes
    for i in range(n_coils, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")

    if title:
        plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_magnitude_phase(image, title_prefix=""):
    """Plot magnitude and phase of a complex image side by side.

    Parameters
    ----------
    image : np.ndarray
        2D complex-valued image.
    title_prefix : str
        Prefix for subplot titles.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(np.abs(image), cmap="gray")
    ax1.set_title(f"{title_prefix} Magnitude")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(np.angle(image), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f"{title_prefix} Phase")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    plt.tight_layout()
    plt.show()


def plot_butterworth_kspace(H, kspace_filtered, kspace_original, D0=30, n=2):
    """Plot the Butterworth filter, filtered k-space, and original k-space.

    Parameters
    ----------
    H : np.ndarray
        2D Butterworth filter mask.
    kspace_filtered : np.ndarray
        Filtered k-space data (complex).
    kspace_original : np.ndarray
        Original k-space data (complex).
    D0 : float
        Cut-off frequency used (for the title).
    n : int
        Filter order used (for the title).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(H, cmap="gray")
    axes[0].set_title(f"Butterworth Filter (D0={D0}, n={n})")
    axes[0].axis("off")
    fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046)

    axes[1].imshow(np.log1p(np.abs(kspace_filtered)), cmap="gray")
    axes[1].set_title("Filtered K-space (log-scaled)")
    axes[1].axis("off")
    fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)

    axes[2].imshow(np.log1p(np.abs(kspace_original)), cmap="gray")
    axes[2].set_title("Original K-space (log-scaled)")
    axes[2].axis("off")
    fig.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046)

    plt.suptitle("Butterworth Low-pass Filter", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
