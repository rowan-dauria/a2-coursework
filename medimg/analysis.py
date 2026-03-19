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


def reconstruct_fbp(sinogram, theta, filter_name="ramp"):
    """Reconstruct an image from a sinogram using filtered back-projection.

    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram (detector positions x angles).
    theta : np.ndarray
        Projection angles in degrees.
    filter_name : str
        Filter for frequency-domain filtering. One of ``'ramp'``,
        ``'shepp-logan'``, ``'cosine'``, ``'hamming'``, ``'hann'``,
        or ``None`` for no filter.

    Returns
    -------
    np.ndarray
        Reconstructed image.
    """
    return iradon(sinogram, theta=theta, circle=False, filter_name=filter_name)


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
        # Division by n_angles is to normalise the learning rate for different number of angles
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


def reconstruct_os_sart(sinogram, theta, image_shape, n_iter=50,
                         n_subsets=10, lr=0.01, profile=False):
    """Reconstruct using Ordered Subsets SART (OS-SART).

    Instead of using all projections per iteration (as SIRT/GD does),
    projections are split into *n_subsets* mini-batches.  The image is
    updated once per subset, giving *n_subsets* updates per iteration.

    Parameters
    ----------
    sinogram : np.ndarray
        The measured sinogram (detector positions x angles).
    theta : np.ndarray
        Projection angles in degrees.
    image_shape : tuple
        Shape of the image to reconstruct.
    n_iter : int
        Number of full iterations (each iterates over all subsets).
    n_subsets : int
        Number of ordered subsets to split projections into.
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

    # Split angle indices into ordered subsets
    subset_indices = np.array_split(np.arange(n_angles), n_subsets)

    t_radon_total = 0.0
    t_iradon_total = 0.0
    t_arith_total = 0.0

    for _ in range(n_iter):
        for idx in subset_indices:
            theta_sub = theta[idx]
            sino_sub = sinogram[:, idx]
            n_sub = len(idx)

            # Forward project current estimate with subset angles
            t0 = time.time()
            sino_est = radon(recon, theta=theta_sub, circle=False)
            t_radon_total += time.time() - t0

            # Residual
            t0 = time.time()
            residual = sino_est - sino_sub
            t_arith_total += time.time() - t0

            # Back-project residual
            t0 = time.time()
            gradient = iradon(
                residual, theta=theta_sub, circle=False, filter_name=None
            )
            t_iradon_total += time.time() - t0

            # Update (normalise by subset size)
            t0 = time.time()
            recon -= lr * gradient / n_sub
            t_arith_total += time.time() - t0

    if profile:
        total = t_radon_total + t_iradon_total + t_arith_total
        print(f"    OS-SART profile ({n_iter} iters, {n_subsets} subsets):")
        print(f"      radon (fwd):   {t_radon_total:.2f}s "
              f"({100 * t_radon_total / total:.0f}%)")
        print(f"      iradon (bkp):  {t_iradon_total:.2f}s "
              f"({100 * t_iradon_total / total:.0f}%)")
        print(f"      arithmetic:    {t_arith_total:.2f}s "
              f"({100 * t_arith_total / total:.0f}%)")
        print(f"      total:         {total:.2f}s")

    return recon


def kspace_to_image(kspace_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D k-space array to image space via inverse FFT.

    Applies ``ifftshift``, 2D inverse FFT, and ``fftshift`` to centre
    the resulting image.

    Parameters
    ----------
    kspace_2d : np.ndarray
        2D complex-valued k-space data.

    Returns
    -------
    np.ndarray
        Complex-valued image.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_2d)))


def combine_coils_rss(images: np.ndarray) -> np.ndarray:
    """Combine multi-coil images using root-sum-of-squares.

    Parameters
    ----------
    images : np.ndarray
        Complex or real array of shape ``(n_coils, H, W)``.

    Returns
    -------
    np.ndarray
        Combined magnitude image of shape ``(H, W)``.
    """
    return np.sqrt(np.sum(np.abs(images) ** 2, axis=0))


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
