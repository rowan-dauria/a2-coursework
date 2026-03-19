"""Tests for medimg.data module."""

import numpy as np
import pytest

from medimg.data import load_ct_image, load_kspace


class TestLoadCtImage:
    """Tests for load_ct_image."""

    def test_shape_and_dtype(self, ct_image_path):
        """Loaded CT image is 2D float with values in [0, 1]."""
        img = load_ct_image(ct_image_path)
        assert img.ndim == 2
        assert np.issubdtype(img.dtype, np.floating)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_nonexistent_raises(self, tmp_path):
        """FileNotFoundError on a missing file."""
        with pytest.raises(FileNotFoundError):
            load_ct_image(str(tmp_path / "does_not_exist.png"))


class TestLoadKspace:
    """Tests for load_kspace."""

    def test_shape_and_dtype(self, kspace_path):
        """Loaded k-space has expected shape and complex dtype."""
        data = load_kspace(kspace_path)
        assert data.shape == (6, 280, 280)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_nonexistent_raises(self, tmp_path):
        """FileNotFoundError on a missing file."""
        with pytest.raises(FileNotFoundError):
            load_kspace(str(tmp_path / "missing.npy"))

    def test_not_empty(self, kspace_path):
        """K-space data contains non-zero values."""
        data = load_kspace(kspace_path)
        assert np.any(data != 0)
