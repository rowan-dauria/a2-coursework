"""Data loading and management utilities."""

import numpy as np
from skimage.io import imread


def load_ct_image(path: str) -> np.ndarray:
    """Load a CT image as a grayscale float array.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Grayscale image with values in [0, 1].
    """
    return imread(path, as_gray=True)


def load_kspace(path: str) -> np.ndarray:
    """Load MRI k-space data from a .npy file.

    Parameters
    ----------
    path : str
        Path to the .npy file.

    Returns
    -------
    np.ndarray
        Complex-valued k-space array.
    """
    return np.load(path)
