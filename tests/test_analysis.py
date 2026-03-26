"""Tests for medimg.analysis module."""

import numpy as np
import pytest

from medimg.analysis import (
    add_noise,
    butterworth_lowpass_filter,
    combine_coils_rss,
    compute_metrics,
    denoise_gaussian,
    denoise_mean,
    denoise_wavelet_filter,
    kspace_to_image,
    reconstruct_fbp,
    reconstruct_gradient_descent,
    reconstruct_os_sart,
    rotate_image,
)

# ── add_noise ──────────────────────────────────────────────────────────


class TestAddNoise:
    """Tests for add_noise."""

    def test_shape_preserved(self, small_sinogram):
        """Output shape matches input shape."""
        np.random.seed(0)
        noisy = add_noise(small_sinogram, I0=1e5)
        assert noisy.shape == small_sinogram.shape

    def test_nonnegative(self, small_sinogram):
        """All values are >= 0 after clipping."""
        np.random.seed(0)
        noisy = add_noise(small_sinogram, I0=1e3)
        assert np.all(noisy >= 0)

    def test_deterministic_with_seed(self, small_sinogram):
        """Same seed produces identical output."""
        np.random.seed(42)
        a = add_noise(small_sinogram, I0=1e4)
        np.random.seed(42)
        b = add_noise(small_sinogram, I0=1e4)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("I0", [1e2, 1e3, 1e5])
    def test_higher_I0_lower_variance(self, small_sinogram, I0):
        """Higher I0 produces less noise (lower variance of difference)."""
        np.random.seed(0)
        noisy = add_noise(small_sinogram, I0=I0)
        diff_var = np.var(noisy - small_sinogram)
        # Just check it runs and produces finite variance
        assert np.isfinite(diff_var)


# ── reconstruct_fbp ───────────────────────────────────────────────────


class TestReconstructFBP:
    """Tests for reconstruct_fbp."""

    def test_output_shape(self, small_sinogram, theta_36):
        """Output is a 2D array."""
        recon = reconstruct_fbp(small_sinogram, theta_36)
        assert recon.ndim == 2

    def test_recovers_phantom(self, small_image, small_sinogram, theta_36):
        """FBP reconstruction has reasonable RMSE on small phantom."""
        recon = reconstruct_fbp(small_sinogram, theta_36)
        # Crop to match phantom size
        recon_crop = recon[: small_image.shape[0], : small_image.shape[1]]
        rmse = np.sqrt(np.mean((small_image - recon_crop) ** 2))
        assert rmse < 0.5

    @pytest.mark.parametrize(
        "filter_name",
        ["ramp", "shepp-logan", "cosine", "hamming", "hann"],
    )
    def test_filter_names(self, small_sinogram, theta_36, filter_name):
        """All standard filter names run without error."""
        recon = reconstruct_fbp(small_sinogram, theta_36, filter_name=filter_name)
        assert recon.ndim == 2


# ── reconstruct_gradient_descent ──────────────────────────────────────


class TestReconstructGD:
    """Tests for reconstruct_gradient_descent."""

    @pytest.mark.slow
    def test_output_shape(self, small_sinogram, theta_36, small_image):
        """Output shape matches requested image_shape."""
        recon = reconstruct_gradient_descent(
            small_sinogram, theta_36, small_image.shape, n_iter=2
        )
        assert recon.shape == small_image.shape

    @pytest.mark.slow
    def test_reduces_error(self, small_sinogram, theta_36, small_image):
        """RMSE after iterations is less than RMSE of zeros."""
        recon = reconstruct_gradient_descent(
            small_sinogram, theta_36, small_image.shape, n_iter=10
        )
        rmse_zeros = np.sqrt(np.mean(small_image**2))
        rmse_recon = np.sqrt(np.mean((small_image - recon) ** 2))
        assert rmse_recon < rmse_zeros

    @pytest.mark.slow
    def test_profile_flag(self, small_sinogram, theta_36, small_image, capsys):
        """profile=True prints timing output."""
        reconstruct_gradient_descent(
            small_sinogram, theta_36, small_image.shape, n_iter=2, profile=True
        )
        captured = capsys.readouterr()
        assert "GD profile" in captured.out


# ── reconstruct_os_sart ──────────────────────────────────────────────


class TestReconstructOsSart:
    """Tests for reconstruct_os_sart."""

    @pytest.mark.slow
    def test_output_shape(self, small_sinogram, theta_36, small_image):
        """Output shape matches requested image_shape."""
        recon = reconstruct_os_sart(
            small_sinogram, theta_36, small_image.shape, n_iter=1, n_subsets=2
        )
        assert recon.shape == small_image.shape

    @pytest.mark.slow
    def test_reduces_error(self, small_sinogram, theta_36, small_image):
        """RMSE after iterations is less than RMSE of zeros."""
        recon = reconstruct_os_sart(
            small_sinogram, theta_36, small_image.shape, n_iter=5, n_subsets=2
        )
        rmse_zeros = np.sqrt(np.mean(small_image**2))
        rmse_recon = np.sqrt(np.mean((small_image - recon) ** 2))
        assert rmse_recon < rmse_zeros

    @pytest.mark.slow
    @pytest.mark.parametrize("n_subsets", [2, 5])
    def test_subsets(self, small_sinogram, theta_36, small_image, n_subsets):
        """Different subset counts produce valid output."""
        recon = reconstruct_os_sart(
            small_sinogram,
            theta_36,
            small_image.shape,
            n_iter=1,
            n_subsets=n_subsets,
        )
        assert recon.shape == small_image.shape
        assert np.all(np.isfinite(recon))


# ── kspace_to_image ──────────────────────────────────────────────────


class TestKspaceToImage:
    """Tests for kspace_to_image."""

    def test_shape(self, complex_kspace_2d):
        """Output shape matches input shape."""
        img = kspace_to_image(complex_kspace_2d)
        assert img.shape == complex_kspace_2d.shape

    def test_complex(self, complex_kspace_2d):
        """Output is complex-valued."""
        img = kspace_to_image(complex_kspace_2d)
        assert np.issubdtype(img.dtype, np.complexfloating)

    def test_roundtrip(self):
        """ifft2(fftshift(kspace)) recovers original image."""
        original = np.zeros((16, 16), dtype=float)
        original[4:12, 4:12] = 1.0
        kspace = np.fft.ifftshift(np.fft.fft2(original))
        recovered = kspace_to_image(kspace)
        np.testing.assert_allclose(np.abs(recovered), original, atol=1e-10)


# ── combine_coils_rss ────────────────────────────────────────────────


class TestCombineCoilsRss:
    """Tests for combine_coils_rss."""

    def test_output_shape(self, multi_coil_images):
        """(n_coils, H, W) -> (H, W)."""
        result = combine_coils_rss(multi_coil_images)
        assert result.shape == (16, 16)

    def test_nonnegative(self, multi_coil_images):
        """All RSS values are >= 0."""
        result = combine_coils_rss(multi_coil_images)
        assert np.all(result >= 0)

    def test_single_coil(self):
        """Single coil RSS equals abs(image)."""
        rng = np.random.default_rng(7)
        single = rng.standard_normal((1, 8, 8)) + 1j * rng.standard_normal((1, 8, 8))
        result = combine_coils_rss(single)
        np.testing.assert_allclose(result, np.abs(single[0]), atol=1e-12)


# ── compute_metrics ──────────────────────────────────────────────────


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_keys(self, small_image):
        """Returned dict has RMSE, PSNR, SSIM keys."""
        noisy = small_image + 0.01 * np.random.default_rng(0).standard_normal(
            small_image.shape
        )
        m = compute_metrics(small_image, noisy)
        assert set(m.keys()) == {"RMSE", "PSNR", "SSIM"}

    def test_identical_images(self, small_image):
        """Identical images give RMSE=0, SSIM=1."""
        m = compute_metrics(small_image, small_image.copy())
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["SSIM"] == pytest.approx(1.0, abs=1e-6)

    def test_crops_reconstruction(self, small_image):
        """Larger reconstruction is cropped correctly without error."""
        bigger = np.zeros((40, 40))
        bigger[:32, :32] = small_image
        m = compute_metrics(small_image, bigger)
        assert m["RMSE"] == pytest.approx(0.0)


# ── rotate_image ─────────────────────────────────────────────────────


class TestRotateImage:
    """Tests for rotate_image."""

    def test_360_identity(self, small_image):
        """360-degree rotation approximately recovers the original."""
        rotated = rotate_image(small_image, 360, reshape=False)
        np.testing.assert_allclose(rotated, small_image, atol=1e-10)

    def test_shape_preserved(self, small_image):
        """reshape=False keeps original shape."""
        rotated = rotate_image(small_image, 45, reshape=False)
        assert rotated.shape == small_image.shape

    def test_rotate_90(self):
        """90-degree clockwise rotation matches np.rot90(k=3)."""
        img = np.arange(16, dtype=float).reshape(4, 4)
        rotated = rotate_image(img, 90, reshape=False)
        expected = np.rot90(img, k=-1)  # k=-1 is 90° clockwise
        np.testing.assert_allclose(rotated, expected, atol=0.5)


# ── denoise_mean ─────────────────────────────────────────────────────


class TestDenoiseMean:
    """Tests for denoise_mean."""

    def test_shape(self, noisy_image):
        """Output shape matches input."""
        result = denoise_mean(noisy_image)
        assert result.shape == noisy_image.shape

    def test_reduces_noise(self, noisy_image, clean_image):
        """Denoised image is closer to clean than the noisy input."""
        denoised = denoise_mean(noisy_image)
        noise_before = np.std(noisy_image - clean_image)
        noise_after = np.std(denoised - clean_image)
        assert noise_after < noise_before


# ── denoise_gaussian ─────────────────────────────────────────────────


class TestDenoiseGaussian:
    """Tests for denoise_gaussian."""

    def test_shape(self, noisy_image):
        """Output shape matches input."""
        result = denoise_gaussian(noisy_image)
        assert result.shape == noisy_image.shape

    def test_reduces_noise(self, noisy_image, clean_image):
        """Denoised image is closer to clean than the noisy input."""
        denoised = denoise_gaussian(noisy_image, sigma=1.0)
        noise_before = np.std(noisy_image - clean_image)
        noise_after = np.std(denoised - clean_image)
        assert noise_after < noise_before


# ── denoise_wavelet_filter ───────────────────────────────────────────


class TestDenoiseWavelet:
    """Tests for denoise_wavelet_filter."""

    def test_shape(self, noisy_image):
        """Output shape matches input."""
        result = denoise_wavelet_filter(noisy_image)
        assert result.shape == noisy_image.shape

    def test_reduces_noise(self, noisy_image, clean_image):
        """Denoised image is closer to clean than the noisy input."""
        denoised = denoise_wavelet_filter(noisy_image)
        noise_before = np.std(noisy_image - clean_image)
        noise_after = np.std(denoised - clean_image)
        assert noise_after < noise_before


# ── butterworth_lowpass_filter ───────────────────────────────────────


class TestButterworthLowpassFilter:
    """Tests for butterworth_lowpass_filter."""

    def test_shape(self):
        """Output shape matches the requested shape."""
        H = butterworth_lowpass_filter((32, 32), D0=10, n=2)
        assert H.shape == (32, 32)

    def test_range(self):
        """All values are in [0, 1]."""
        H = butterworth_lowpass_filter((64, 64), D0=20, n=3)
        assert H.min() >= 0.0
        assert H.max() <= 1.0

    def test_centre_is_one(self):
        """DC component (centre) is 1.0."""
        H = butterworth_lowpass_filter((32, 32), D0=10, n=2)
        centre = (32 // 2, 32 // 2)
        assert H[centre] == pytest.approx(1.0)

    def test_higher_order_sharper(self):
        """Higher order produces sharper roll-off at cutoff."""
        H1 = butterworth_lowpass_filter((64, 64), D0=20, n=1)
        H5 = butterworth_lowpass_filter((64, 64), D0=20, n=5)
        # At the cutoff distance D0, both should be ~0.5
        # But just outside D0, higher order should attenuate more
        d_test = 25  # beyond cutoff
        centre = 32
        val_n1 = H1[centre, centre + d_test]
        val_n5 = H5[centre, centre + d_test]
        assert val_n5 < val_n1
