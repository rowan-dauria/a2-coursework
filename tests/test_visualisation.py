"""Tests for medimg.visualisation module."""

import numpy as np

from medimg.visualisation import (
    plot_butterworth_kspace,
    plot_coil_grid,
    plot_magnitude_phase,
    plot_reconstruction_comparison,
    plot_sinogram_grid,
    print_metrics_table,
)


class TestPlotSinogramGrid:
    """Tests for plot_sinogram_grid."""

    def test_runs(self):
        """Function executes without error on minimal input."""
        sino = np.random.default_rng(0).standard_normal((32, 10))
        sino_dict = {36: {1e5: sino, 1e3: sino}}
        plot_sinogram_grid(sino_dict, row_params=[36], I0_levels=[1e5, 1e3])


class TestPlotReconstructionComparison:
    """Tests for plot_reconstruction_comparison."""

    def test_runs(self):
        """Function executes without error on minimal input."""
        img = np.random.default_rng(0).standard_normal((16, 16))
        metrics = {"RMSE": 0.1, "PSNR": 30.0, "SSIM": 0.9}
        results = {
            36: {
                1e5: {
                    "fbp": {"image": img, "metrics": metrics},
                    "gd": {"image": img, "metrics": metrics},
                }
            }
        }
        plot_reconstruction_comparison(
            results, row_params=[36], I0_levels=[1e5]
        )


class TestPrintMetricsTable:
    """Tests for print_metrics_table."""

    def test_prints_output(self, capsys):
        """Function prints a table to stdout."""
        metrics = {"RMSE": 0.1, "PSNR": 30.0, "SSIM": 0.9}
        results = {
            36: {
                1e5: {
                    "fbp": {"metrics": metrics},
                    "gd": {"metrics": metrics},
                }
            }
        }
        print_metrics_table(results, row_params=[36], I0_levels=[1e5])
        captured = capsys.readouterr()
        assert "RMSE" in captured.out
        assert "PSNR" in captured.out


class TestPlotCoilGrid:
    """Tests for plot_coil_grid."""

    def test_runs(self):
        """Function executes without error on small input."""
        images = np.random.default_rng(0).standard_normal((4, 8, 8))
        plot_coil_grid(images, n_coils=4)


class TestPlotMagnitudePhase:
    """Tests for plot_magnitude_phase."""

    def test_runs(self):
        """Function executes without error on small complex input."""
        rng = np.random.default_rng(0)
        img = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        plot_magnitude_phase(img)


class TestPlotButterworthKspace:
    """Tests for plot_butterworth_kspace."""

    def test_runs(self):
        """Function executes without error on small input."""
        rng = np.random.default_rng(0)
        H = np.ones((16, 16))
        kspace = rng.standard_normal((16, 16)) + 1j * rng.standard_normal(
            (16, 16)
        )
        plot_butterworth_kspace(H, kspace, kspace)
