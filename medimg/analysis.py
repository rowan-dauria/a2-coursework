"""Reconstruction, noise simulation, and image quality metrics."""

import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import iradon, radon


def add_noise(sinogram, I0, sigma=0.05):
    """Apply Poisson and Gaussian noise to a sinogram.

    Simulates photon-counting noise via the Beer-Lambert law: the clean
    absorption sinogram is converted to transmission intensities, Poisson
    and Gaussian noise are added in the transmission domain, and the
    result is converted back to absorption values.

    Parameters
    ----------
    sinogram : np.ndarray
        Sinogram with realistic attenuation values (line integrals of mu).
    I0 : float
        Source intensity for Poisson noise (photon count).
    sigma : float
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    np.ndarray
        Noisy sinogram (same scale as input).
    """
    # Beer-Lambert: convert absorption sinogram to transmission intensities
    I_transmitted = I0 * np.exp(-sinogram)

    # Poisson noise on photon counts
    I_noisy = np.random.poisson(I_transmitted).astype(float)

    # Additive Gaussian noise on the detector signal (transmission domain)
    I_noisy += np.random.normal(loc=0.0, scale=sigma, size=I_noisy.shape)

    # Clip to at least 1 photon to avoid log(0)
    I_noisy = np.clip(I_noisy, 1, None)

    # Convert back to absorption sinogram
    sinogram_noisy = -np.log(I_noisy / I0)

    # Clip negative attenuation values (non-physical)
    sinogram_noisy = np.clip(sinogram_noisy, 0, None)

    return sinogram_noisy


def reconstruct_fbp(sinogram, theta):
    """Reconstruct an image from a sinogram using filtered back-projection.

    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram (detector positions x angles).
    theta : np.ndarray
        Projection angles in degrees.

    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    return iradon(sinogram, theta=theta, circle=False)


def reconstruct_gradient_descent(sinogram, theta, image_shape, n_iter=50,
                                 lr=0.01, profile=False):
    """Reconstruct an image from a sinogram using gradient descent (SIRT-like).

    Minimises ``||Ax - b||^2`` where *A* is the Radon transform and *b* is the
    sinogram.

    Parameters
    ----------
    sinogram : np.ndarray
        The measured sinogram.
    theta : np.ndarray
        Projection angles in degrees.
    image_shape : tuple
        Shape of the image to reconstruct.
    n_iter : int
        Number of gradient descent iterations.
    lr : float
        Learning rate (step size).
    profile : bool
        If True, print per-step timing breakdown.

    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    recon = np.zeros(image_shape)
    n_angles = len(theta)

    t_radon_total = 0.0
    t_iradon_total = 0.0
    t_arith_total = 0.0

    for i in range(n_iter):
        # Forward project current estimate
        t0 = time.time()
        sino_est = radon(recon, theta=theta, circle=False)
        t_radon_total += time.time() - t0

        # Residual in sinogram space
        t0 = time.time()
        residual = sino_est - sinogram
        t_arith_total += time.time() - t0

        # Back-project the residual (gradient of ||Ax - b||^2 w.r.t. x)
        t0 = time.time()
        gradient = iradon(residual, theta=theta, circle=False, filter_name=None)
        t_iradon_total += time.time() - t0

        # Update
        t0 = time.time()
        recon -= lr * gradient / n_angles
        t_arith_total += time.time() - t0

    if profile:
        total = t_radon_total + t_iradon_total + t_arith_total
        print(f"    GD profile ({n_iter} iters):")
        print(f"      radon (fwd):   {t_radon_total:.2f}s "
              f"({100 * t_radon_total / total:.0f}%)")
        print(f"      iradon (bkp):  {t_iradon_total:.2f}s "
              f"({100 * t_iradon_total / total:.0f}%)")
        print(f"      arithmetic:    {t_arith_total:.2f}s "
              f"({100 * t_arith_total / total:.0f}%)")
        print(f"      total:         {total:.2f}s")

    return recon


def compute_metrics(ground_truth, reconstruction):
    """Compute RMSE, PSNR, and SSIM between ground truth and reconstruction.

    Parameters
    ----------
    ground_truth : np.ndarray
        The reference image.
    reconstruction : np.ndarray
        The reconstructed image.

    Returns
    -------
    dict
        Dictionary with ``'RMSE'``, ``'PSNR'``, and ``'SSIM'`` keys.
    """
    # Crop reconstruction to match ground truth if needed
    recon_cropped = reconstruction[
        :ground_truth.shape[0], :ground_truth.shape[1]
    ]

    data_range = ground_truth.max() - ground_truth.min()
    rmse = np.sqrt(np.mean((ground_truth - recon_cropped) ** 2))
    psnr_val = psnr(ground_truth, recon_cropped, data_range=data_range)
    ssim_val = ssim(
        ground_truth, recon_cropped, data_range=data_range, win_size=7
    )

    return {"RMSE": rmse, "PSNR": psnr_val, "SSIM": ssim_val}
