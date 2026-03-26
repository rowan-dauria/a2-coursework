"""Shared fixtures for medimg test suite."""

import os

import matplotlib
import numpy as np
import pytest
from skimage.transform import radon

# Use non-interactive backend to avoid GUI windows during tests
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def small_image():
    """32x32 float phantom with a filled circle."""
    img = np.zeros((32, 32), dtype=float)
    y, x = np.ogrid[-16:16, -16:16]
    mask = x**2 + y**2 <= 10**2
    img[mask] = 1.0
    return img


@pytest.fixture(scope="session")
def theta_36():
    """36 projection angles spanning [0, 180)."""
    return np.linspace(0, 180, 36, endpoint=False)


@pytest.fixture(scope="session")
def small_sinogram(small_image, theta_36):
    """Radon transform of `small_image` at 36 angles."""
    return radon(small_image, theta=theta_36, circle=False)


@pytest.fixture(scope="session")
def complex_kspace_2d():
    """16x16 complex k-space (DC centred via ifftshift)."""
    img = np.zeros((16, 16), dtype=float)
    img[4:12, 4:12] = 1.0
    return np.fft.ifftshift(np.fft.fft2(img))


@pytest.fixture(scope="session")
def multi_coil_images():
    """Shape (4, 16, 16) complex array simulating 4 coil images."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 16, 16)) + 1j * rng.standard_normal((4, 16, 16))


@pytest.fixture(scope="session")
def noisy_image():
    """64x64 float image with additive Gaussian noise."""
    rng = np.random.default_rng(0)
    clean = np.zeros((64, 64), dtype=float)
    clean[16:48, 16:48] = 1.0
    return clean + 0.1 * rng.standard_normal((64, 64))


@pytest.fixture(scope="session")
def clean_image():
    """64x64 clean float image (matching noisy_image ground truth)."""
    clean = np.zeros((64, 64), dtype=float)
    clean[16:48, 16:48] = 1.0
    return clean


@pytest.fixture(scope="session")
def ct_image_path():
    """Path to the CT image file."""
    return os.path.join(
        os.path.dirname(__file__), os.pardir, "data", "CT_exercise_1.png"
    )


@pytest.fixture(scope="session")
def kspace_path():
    """Path to the MRI k-space file."""
    return os.path.join(os.path.dirname(__file__), os.pardir, "data", "knee.npy")
